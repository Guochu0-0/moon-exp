function traditional270_split(method, repo, in_dir, out_dir, n_threads)
% #270: official feature calls, no FSC/CSC geometric filtering.
% Run ONE MATLAB process, no auto parpool. Inputs are mapped floating images.
% Output p1/p2: MATLAB 1-based xy, N x 2; sec and err.
if nargin < 5, n_threads = 2; end
maxNumCompThreads(n_threads);
ps = parallel.Settings; ps.Pool.AutoCreate = false;
addpath(repo);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
files = dir(fullfile(in_dir, '*.mat'));
[~, order] = sort({files.name}); files = files(order);
for i = 1:numel(files)
    dest = fullfile(out_dir, files(i).name);
    if exist(dest, 'file'), continue; end
    S = load(fullfile(in_dir, files(i).name));
    p1 = zeros(0,2); p2 = zeros(0,2); err = ''; t = tic;
    try
        rng(0);
        switch method
            case 'rift'
                [d1,d2] = RIFT_no_rotation_invariance(repmat(double(S.opt),[1 1 3]),...
                    repmat(double(S.sar),[1 1 3]),4,6,96);
                idx = matchFeatures(d1.des,d2.des,'MaxRatio',1,'MatchThreshold',100);
                p1 = d1.kps(idx(:,1),1:2); p2 = d2.kps(idx(:,2),1:2);
                [p2, keep] = unique(p2,'rows'); p1 = p1(keep,:);
            case 'ossift'
                a = imadjust(double(S.opt)) + 0.001;
                b = imadjust(double(S.sar)) + 0.001;
                [h1,g1,o1] = build_scale_opt(a,2,8,2^(1/3),0.04);
                [h2,g2,o2] = build_scale_sar(b,2,8,2^(1/3),0.04);
                k1 = find_scale_extreme(h1,0.00001,2,2^(1/3),g1,o1);
                k2 = find_scale_extreme(h2,0.00001,2,2^(1/3),g2,o2);
                [d1,l1] = calc_descriptors_parallel(g1,o1,k1);
                [d2,l2] = calc_descriptors_parallel(g2,o2,k2);
                % Exact first descriptor-only stage of official CSC2.m.
                % Follow its call order: SAR queries, optical database.
                if size(d1,1) >= 2
                    for j = 1:size(d2,1)
                        [vals,idx] = sort(acos(d2(j,:) * d1'));
                        if vals(1) < 0.95 * vals(2)
                            p1(end+1,:) = l1(idx(1),1:2); %#ok<AGROW>
                            p2(end+1,:) = l2(j,1:2); %#ok<AGROW>
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
    tmp = [dest '.tmp.mat']; save(tmp,'p1','p2','sec','err'); movefile(tmp,dest);
    fprintf('%s %d/%d n=%d sec=%.2f %s\n',method,i,numel(files),size(p1,1),sec,err);
end
end
