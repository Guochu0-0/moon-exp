function traditional270_split(method, repo, in_dir, out_dir, n_threads)
% #270: official feature calls, no FSC/CSC geometric filtering.
% Run ONE MATLAB process, no auto parpool. Inputs are mapped floating images.
% Output p1/p2: MATLAB 1-based xy, N x 2; conf, sec and err.
if nargin < 5, n_threads = 2; end
previous_threads = maxNumCompThreads(n_threads);
thread_cleanup = onCleanup(@() maxNumCompThreads(previous_threads));
ps = parallel.Settings;
previous_pool = ps.Pool.AutoCreate;
ps.Pool.AutoCreate = false;
pool_cleanup = onCleanup(@() restore_pool(ps, previous_pool));
manifest = jsondecode(fileread(fullfile(fileparts(in_dir), 'manifest.json')));
cfg = manifest.config;
assert(strcmp(cfg.method, method), 'Method differs from input manifest.');
params = cfg.params;
method_path = fullfile(repo, cfg.repo);
addpath(method_path);
path_cleanup = onCleanup(@() rmpath(method_path));
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
files = dir(fullfile(in_dir, '*.mat'));
[~, order] = sort({files.name}); files = files(order);
for i = 1:numel(files)
    dest = fullfile(out_dir, files(i).name);
    if exist(dest, 'file')
        previous = load(dest, 'err');
        if isempty(previous.err), continue; end
    end
    S = load(fullfile(in_dir, files(i).name));
    p1 = zeros(0,2); p2 = zeros(0,2); conf = zeros(0,1); err = ''; t = tic;
    try
        rng(cfg.seed);
        switch method
            case 'rift'
                [d1,d2] = RIFT_no_rotation_invariance(repmat(double(S.opt),[1 1 3]),...
                    repmat(double(S.sar),[1 1 3]),params.scales,params.orientations,params.patch_size);
                if ~isempty(d1.des) && ~isempty(d2.des)
                    [idx, distance] = matchFeatures(d1.des, d2.des, ...
                        'MaxRatio', params.max_ratio, 'MatchThreshold', params.match_threshold);
                    p1 = d1.kps(idx(:,1),1:2); p2 = d2.kps(idx(:,2),1:2);
                    [p2, keep] = unique(p2,'rows'); p1 = p1(keep,:);
                    conf = 1 ./ (1 + distance(keep));
                end
            case 'ossift'
                a = imadjust(double(S.opt)) + 0.001;
                b = imadjust(double(S.sar)) + 0.001;
                [h1,g1,o1] = build_scale_opt(a,params.sigma,params.layers,params.scale_ratio,params.harris_d);
                [h2,g2,o2] = build_scale_sar(b,params.sigma,params.layers,params.scale_ratio,params.harris_d);
                k1 = find_scale_extreme(h1,params.harris_threshold,params.sigma,params.scale_ratio,g1,o1);
                k2 = find_scale_extreme(h2,params.harris_threshold,params.sigma,params.scale_ratio,g2,o2);
                [d1,l1] = calc_descriptors_parallel(g1,o1,k1);
                [d2,l2] = calc_descriptors_parallel(g2,o2,k2);
                % Exact first descriptor-only stage of official CSC2.m.
                % Follow its call order: SAR queries, optical database.
                if size(d1,1) >= 2
                    for j = 1:size(d2,1)
                        [vals,idx] = sort(acos(d2(j,:) * d1'));
                        if vals(1) < params.descriptor_ratio * vals(2)
                            p1(end+1,:) = l1(idx(1),1:2); %#ok<AGROW>
                            p2(end+1,:) = l2(j,1:2); %#ok<AGROW>
                            conf(end+1,1) = 1 - vals(1) / vals(2); %#ok<AGROW>
                        end
                    end
                end
            otherwise
                error('Unsupported method');
        end
    catch e
        err = [e.identifier ': ' e.message];
    end
    sec = toc(t);
    tmp = [dest '.tmp.mat']; save(tmp,'p1','p2','conf','sec','err'); movefile(tmp,dest);
    fprintf('%s %d/%d n=%d sec=%.2f %s\n',method,i,numel(files),size(p1,1),sec,err);
    if ~isempty(err), rethrow(e); end
end
end

function restore_pool(settings, previous)
settings.Pool.AutoCreate = previous;
end
