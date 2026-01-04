files = dir(fullfile('./models', '*.mat'));

threshold = 0.5;

totalCount = 0;
totalElems = 0;

for k = 1:numel(files)

    fname = fullfile('./models', files(k).name);
    data = load(fname);

    model = data.model;

    if model.rmse_min <= threshold
        totalCount = totalCount + 1;
        fprintf("%s | RMSE_TR (min): %.6f | RMSE_VL: %.6f | MEE (min): %.6f \n",...
            files(k).name, model.rmse_min, model.rmse_validation, model.mee_min);
    end

    totalElems = totalElems + 1;
end

fprintf("\nTOTAL: %d / %d (%.2f%%)\n", ...
    totalCount, totalElems, 100 * totalCount / totalElems);
