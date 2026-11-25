"""RIPE++ training losses on the project's stock MINIMA inference modules."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from baselines.adapters.base import to_original
from moonlib.ransac import fit_affine


@torch.no_grad()
def geometry(points0, points1, matches, grid0, grid1, cfg):
    if len(matches) < cfg["min_candidates"]:
        return torch.zeros(len(matches), dtype=torch.bool, device=matches.device), "few_matches"
    p0 = to_original(points0[matches[:, 0]].cpu().numpy(), *grid0)
    p1 = to_original(points1[matches[:, 1]].cpu().numpy(), *grid1)
    matrix, mask, _ = fit_affine(np.column_stack((p0, p1, np.ones(len(matches)))), cfg["threshold_px"])
    if matrix is None or not np.isfinite(matrix).all():
        return torch.zeros(len(matches), dtype=torch.bool, device=matches.device), "fit_failed"
    return torch.as_tensor(mask, device=matches.device), "ok"


def dense_superpoint(model, image):
    for layer in range(1, 5):
        image = model.relu(getattr(model, f"conv{layer}a")(image))
        image = model.relu(getattr(model, f"conv{layer}b")(image))
        if layer < 4:
            image = model.pool(image)
    return model.convPb(model.relu(model.convPa(image))), model.convDb(model.relu(model.convDa(image)))


def sampled_features(model, image, bf16):
    from lightglue.superpoint import sample_descriptors

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        logits, description = checkpoint(dense_superpoint, model, image, use_reentrant=False)
    _, _, height, width = logits.shape
    logits = logits[0].float().permute(1, 2, 0).reshape(-1, 65)
    log_probs = logits.log_softmax(-1)
    position_probs = logits[:, :64].log_softmax(-1)
    with torch.no_grad():
        action = torch.multinomial(log_probs.exp(), 1).squeeze(1)
        selected = action != 64
        # The extractor's descriptor pool includes rejected cells. Draw their
        # latent positions from the distribution conditional on a location.
        position = action.clone()
        position[~selected] = torch.multinomial(position_probs[~selected].exp(), 1).squeeze(1)
        cell = torch.arange(height * width, device=image.device)
        xy = torch.stack((8 * (cell % width) + position % 8,
                          8 * (cell // width) + position // 8), dim=1).float()
    desc = sample_descriptors(xy[None], F.normalize(description.float(), dim=1))[0].T.contiguous()
    entropy = -(position_probs[selected].exp() * position_probs[selected]).sum()
    return xy, log_probs.gather(1, action[:, None]).squeeze(1), selected, desc, entropy


@torch.no_grad()
def mutual_neighbors(desc0, desc1):
    """Bound peak matrix size while retaining global mutual nearest neighbors."""
    candidates0, values1, nearest1 = [], None, None
    for start in range(0, len(desc0), 512):
        distance = torch.cdist(desc0[start:start + 512], desc1)
        candidates0.append(distance.argmin(1))
        values, indices = distance.min(0)
        if values1 is None:
            values1, nearest1 = values, indices + start
        else:
            nearest1 = torch.where(values < values1, indices + start, nearest1)
            values1 = torch.minimum(values1, values)
    nearest0 = torch.cat(candidates0)
    rows = torch.arange(len(nearest0), device=desc0.device)
    mutual = nearest1[nearest0] == rows
    return torch.stack((rows[mutual], nearest0[mutual]), dim=1)


@torch.no_grad()
def second_neighbors(queries, pool):
    return torch.cat([torch.cdist(chunk, pool).topk(2, dim=1, largest=False).indices[:, 1]
                      for chunk in queries.split(256)])


def descriptor_loss(desc0, desc1, matches, inliers, cfg):
    fallback = int(inliers.sum()) < cfg["descriptor_fallback_below"]
    pairs = matches if fallback else matches[inliers]
    if not len(pairs):
        return (desc0.sum() + desc1.sum()) * 0, fallback
    pos0, pos1 = desc0[pairs[:, 0]], desc1[pairs[:, 1]]
    neg1 = second_neighbors(pos0, desc1)
    neg0 = second_neighbors(pos1, desc0)
    hard = torch.minimum(torch.linalg.vector_norm(pos0 - desc1[neg1], dim=1),
                         torch.linalg.vector_norm(pos1 - desc0[neg0], dim=1))
    loss = (cfg["descriptor_margin"] + F.pairwise_distance(pos0, pos1) - hard).relu().mean()
    return loss, fallback


def extractor_pair(features0, features1, grid0, grid1, cfg, ramp):
    xy0, log0, selected0, desc0, entropy0 = features0
    xy1, log1, selected1, desc1, entropy1 = features1
    ids0, ids1 = selected0.nonzero().flatten(), selected1.nonzero().flatten()
    counts = (len(ids0), len(ids1))
    entropy = (entropy0, entropy1, *counts)
    row = {"keypoints": counts, "candidates": 0, "inliers": 0, "geometry": "few_keypoints", "fallback": None}
    if min(counts) < cfg["min_keypoints"]:
        return None, entropy, row
    rel = mutual_neighbors(desc0[ids0], desc1[ids1])
    matches = torch.stack((ids0[rel[:, 0]], ids1[rel[:, 1]]), dim=1)
    inliers, state = geometry(xy0, xy1, matches, grid0, grid1, cfg)

    background = cfg["background"] * ramp
    rewards = torch.where(inliers, cfg["inlier"], cfg["outlier"])
    background_sum = background * (log0.mean() + log1.mean())
    correction = ((rewards - background) * (log0[matches[:, 0]] + log1[matches[:, 1]])).sum()
    policy = -background_sum - correction / (len(log0) * len(log1))
    count = cfg["keypoint_penalty"] * ramp * (log0[selected0].mean() + log1[selected1].mean())
    desc, fallback = descriptor_loss(desc0, desc1, matches, inliers, cfg)
    row.update(candidates=len(matches), inliers=int(inliers.sum()), geometry=state, fallback=fallback)
    return (policy, count, desc), entropy, row


def match_objective(log_assignment, similarity, matches, inliers, cfg, ramp):
    with torch.no_grad():
        probability = similarity.softmax(2) * similarity.softmax(1)
    weighted = probability * log_assignment[:, :-1, :-1]
    background = cfg["background"] * ramp
    reward = torch.where(inliers, cfg["inlier"], cfg["outlier"])
    correction = (weighted[:, matches[:, 0], matches[:, 1]] * (reward - background)).sum()
    policy = -(weighted.sum() * background + correction)
    dustbins = log_assignment[:, :-1, -1].exp().sum() + log_assignment[:, -1, :-1].exp().sum()
    return policy, cfg["dustbin_penalty"] * dustbins


def matcher_pair(model, features0, features1, grid0, grid1, cfg, ramp):
    from lightglue.lightglue import filter_matches, normalize_keypoints

    xy0, xy1 = features0["keypoints"], features1["keypoints"]
    desc0 = model.input_proj(features0["descriptors"])
    desc1 = model.input_proj(features1["descriptors"])
    if not xy0.shape[1] or not xy1.shape[1]:
        # No cross-image assignment is defined. Retain the nonempty side's
        # unmatched penalty, with zero contribution from an empty side.
        terms = []
        for head in model.log_assignment:
            terms.append(sum((1 - head.get_matchability(desc)).sum() for desc in (desc0, desc1)))
        dustbin = torch.stack(terms).mean() * cfg["dustbin_penalty"]
        return (dustbin * 0, dustbin, dustbin * 0), {
            "keypoints": (xy0.shape[1], xy1.shape[1]), "candidates": 0, "inliers": 0, "geometry": "few_matches"}

    pos0 = model.posenc(normalize_keypoints(xy0, features0["image_size"]))
    pos1 = model.posenc(normalize_keypoints(xy1, features1["image_size"]))
    states = []
    for layer in model.transformers:
        desc0, desc1 = checkpoint(layer, desc0, desc1, pos0, pos1, use_reentrant=False)
        states.append((desc0, desc1))
    with torch.no_grad():
        final, _ = model.log_assignment[-1](desc0, desc1)
        hard, _, _, _ = filter_matches(final, cfg["filter_threshold"])
        selected = (hard[0] >= 0).nonzero().flatten()
        matches = torch.stack((selected, hard[0, selected]), dim=1)
        inliers, state = geometry(xy0[0], xy1[0], matches, grid0, grid1, cfg)
        target0 = final[:, :-1].argmax(-1)
        target1 = final[:, :, :-1].argmax(-2)
        del final

    terms = []
    for i, (encoded0, encoded1) in enumerate(states):
        def losses(a, b, index=i):
            assignment, similarity = model.log_assignment[index](a, b)
            policy, penalty = match_objective(assignment, similarity, matches, inliers, cfg, ramp)
            confidence = policy.new_zeros(())
            if index < len(states) - 1:
                token = model.token_confidence[index].token[0]
                correct0 = assignment[:, :-1].detach().argmax(-1) == target0
                correct1 = assignment[:, :, :-1].detach().argmax(-2) == target1
                side0 = F.binary_cross_entropy_with_logits(token(a.detach()).squeeze(-1), correct0.float())
                side1 = F.binary_cross_entropy_with_logits(token(b.detach()).squeeze(-1), correct1.float())
                confidence = (side0 + side1) * 0.5
            return policy, penalty, confidence

        terms.append(checkpoint(losses, encoded0, encoded1, use_reentrant=False))
    policy, dustbin, confidence = (torch.stack(values).sum() for values in zip(*terms))
    layers = len(terms)
    return (policy / layers, dustbin / layers, confidence / (layers - 1)), {
        "keypoints": (xy0.shape[1], xy1.shape[1]), "candidates": len(matches), "inliers": int(inliers.sum()),
        "geometry": state}
