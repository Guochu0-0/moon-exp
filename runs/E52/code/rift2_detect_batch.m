function rift2_detect_batch(in_dir, out_dir)
% RIFT2 FeatureDetection, returning the detector's FAST Metric before orientation.
n_threads = maxNumCompThreads(2);
restore_threads = onCleanup(@() maxNumCompThreads(n_threads)); %#ok<NASGU>
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
files = dir(fullfile(in_dir, '*.mat'));
[~, order] = sort({files.name}); files = files(order);
for i = 1:numel(files)
    out = fullfile(out_dir, files(i).name);
    if exist(out, 'file'), continue; end
    S = load(fullfile(in_dir, files(i).name));
    optical = detect(S.optical);
    sar = detect(S.sar);
    temporary = [out '.tmp.mat'];
    save(temporary, 'optical', 'sar', '-mat');
    movefile(temporary, out);
    if mod(i, 25) == 0, fprintf('RIFT2 %d/%d\n', i, numel(files)); end
end
end

function result = detect(im)
[m, ~, ~, ~, ~, ~, ~] = phasecong3(double(im), 4, 6, 3, ...
    'mult', 1.6, 'sigmaOnf', 0.75, 'g', 3, 'k', 1);
lo = min(m(:));
hi = max(m(:));
if hi <= lo
    result = zeros(0, 3);
    return
end
points = detectFASTFeatures((m - lo) / (hi - lo), 'MinContrast', 0.0001, 'MinQuality', 0.0001);
points = points.selectStrongest(5000);
result = [double(points.Location) - 1, double(points.Metric)];
end
