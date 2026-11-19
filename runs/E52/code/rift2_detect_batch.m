function rift2_detect_batch(in_dir, out_dir)
% RIFT2 FeatureDetection, returning the detector's FAST Metric before orientation.
maxNumCompThreads(2);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
files = dir(fullfile(in_dir, '*.mat'));
for i = 1:numel(files)
    out = fullfile(out_dir, files(i).name);
    if exist(out, 'file'), continue; end
    S = load(fullfile(in_dir, files(i).name));
    optical = detect(S.optical);
    sar = detect(S.sar);
    temporary = [out '.tmp'];
    save(temporary, 'optical', 'sar', '-mat');
    movefile(temporary, out);
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
