function rift2_split(in_dir, out_dir, seed, n_threads)
% RIFT2 (third_party/RIFT2, 0e980ce) over every <pair>.mat in in_dir; writes <pair>.mat to out_dir.
% Input  .mat: opt, sar  (single H x W, already mapped to [0,1] by baselines/rift2.py).
% Output .mat: p1, p2 (N x 2, MATLAB 1-based, raw matches BEFORE FSC), sec, err.
% Only reorders the official demo_RIFT2.m calls, same parameters. Differences, all deliberate:
%  - no im2uint8 (demo_RIFT2.m:6-7): inputs are float [0,1]; phasecong3 just does double(im)
%  - no FSC / plotting (demo_RIFT2.m:39-53): the unified affine RANSAC runs later in baselines/fit.py
%  - parfor in FeatureDescribe.m runs serially (no pool auto-create) and threads are capped, to keep the
%    shared server lightly loaded
if nargin < 3, seed = 0; end
if nargin < 4, n_threads = 4; end
warning('off', 'all');
maxNumCompThreads(n_threads);
try
    ps = parallel.Settings; ps.Pool.AutoCreate = false;
catch
end
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
files = dir(fullfile(in_dir, '*.mat'));
[~, order] = sort({files.name}); files = files(order);
fprintf('rift2_split: %d pairs in %s\n', numel(files), in_dir);
for i = 1:numel(files)
    out = fullfile(out_dir, files(i).name);
    if exist(out, 'file'), continue; end
    S = load(fullfile(in_dir, files(i).name));
    p1 = zeros(0, 2); p2 = zeros(0, 2); err = ''; t0 = tic;
    try
        rng(seed);
        im1 = gray3(S.opt); im2 = gray3(S.sar);                 % same as demo_RIFT2.m:9-15
        [key1, m1, eo1] = FeatureDetection(im1, 4, 6, 5000);    % demo_RIFT2.m:19-20
        [key2, m2, eo2] = FeatureDetection(im2, 4, 6, 5000);
        kpts1 = kptsOrientation(key1, m1, 1, 96);               % :23-24
        kpts2 = kptsOrientation(key2, m2, 1, 96);
        des1 = FeatureDescribe(im1, eo1, kpts1, 96, 6, 6);      % :27-28
        des2 = FeatureDescribe(im2, eo2, kpts2, 96, 6, 6);
        indexPairs = matchFeatures(des1', des2', 'MaxRatio', 1, 'MatchThreshold', 100);   % :31
        kpts1 = kpts1'; kpts2 = kpts2';
        p1 = kpts1(indexPairs(:, 1), 1:2);
        p2 = kpts2(indexPairs(:, 2), 1:2);
        [p2, IA] = unique(p2, 'rows');                          % :35-36
        p1 = p1(IA, :);
    catch e
        err = [e.identifier ': ' e.message];
    end
    sec = toc(t0);
    save(out, 'p1', 'p2', 'sec', 'err');
    if mod(i, 25) == 0, fprintf('  %d/%d\n', i, numel(files)); end
end
fprintf('rift2_split done\n');
end

function im = gray3(a)
im = double(a);
if size(im, 3) == 1, im = repmat(im, [1 1 3]); end
end
