function scores = evaluate_population(population, objectiveFcn, useParallel)
%EVALUATE_POPULATION 评价一批候选；目标函数约定为越大越好。

populationSize = size(population, 1);
scores = -inf(populationSize, 1);
if useParallel
    parfor i = 1:populationSize
        scores(i) = safe_objective(objectiveFcn, population(i, :));
    end
else
    for i = 1:populationSize
        scores(i) = safe_objective(objectiveFcn, population(i, :));
    end
end
end

function score = safe_objective(objectiveFcn, x)
try
    score = objectiveFcn(x);
    if ~isscalar(score) || ~isfinite(score)
        score = -inf;
    end
catch ME
    warning('SmokeOptimization:CandidateFailed', '候选评价出现异常，已按不可行解处理：%s', ME.message);
    score = -inf;
end
end
