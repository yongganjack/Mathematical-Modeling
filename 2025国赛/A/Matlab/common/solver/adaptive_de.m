function [bestX, bestScore, history, candidates] = adaptive_de(objectiveFcn, lowerBound, upperBound, options, repairFcn)
%ADAPTIVE_DE 成功历史参数、自适应档案和线性种群缩减的差分进化。

if nargin < 5 || isempty(repairFcn)
    repairFcn = @(x) min(upperBound, max(lowerBound, x));
end
dimension = numel(lowerBound);
if numel(upperBound) ~= dimension || any(upperBound <= lowerBound)
    error('SmokeOptimization:InvalidDEBounds', 'DE 上下界维度不一致或存在非正区间。');
end
rng(options.seed, 'twister');

initialPopulationSize = options.populationSize;
population = stratified_population(initialPopulationSize, lowerBound, upperBound);
if isfield(options,'initialPoints') && ~isempty(options.initialPoints)
    initialPoints = options.initialPoints;
    if size(initialPoints,2) ~= dimension
        error('SmokeOptimization:InvalidInitialPoints', 'DE 几何初始点维度与决策变量不一致。');
    end
    initialCount = min(initialPopulationSize,size(initialPoints,1));
    population(1:initialCount,:) = initialPoints(1:initialCount,:);
end
for i = 1:initialPopulationSize
    population(i,:) = repairFcn(population(i,:));
end
scores = evaluate_population(population, objectiveFcn, options.useParallel);
evaluations = initialPopulationSize;
archive = zeros(0, dimension);
memoryF = 0.5 .* ones(1, options.memorySize);
memoryCR = 0.5 .* ones(1, options.memorySize);
memoryIndex = 1;
stallCount = 0;
historyData = zeros(options.maxIterations, 6);
[bestScore, idx] = max(scores);
bestX = population(idx,:);

for iteration = 1:options.maxIterations
    populationSize = size(population, 1);
    [~, ranking] = sort(scores, 'descend');
    pCount = max(2, round(0.11 * populationSize));
    trial = zeros(size(population));
    usedF = zeros(populationSize, 1);
    usedCR = zeros(populationSize, 1);

    combined = [population; archive];
    for i = 1:populationSize
        mem = randi(options.memorySize);
        F = -1;
        while F <= 0
            F = memoryF(mem) + 0.1 * tan(pi * (rand - 0.5));
        end
        F = min(F, 1);
        CR = min(1, max(0, memoryCR(mem) + 0.1 * randn));
        usedF(i) = F;
        usedCR(i) = CR;

        pbest = ranking(randi(pCount));
        r1 = randi(populationSize);
        while r1 == i
            r1 = randi(populationSize);
        end
        r2 = randi(size(combined,1));
        guard = 0;
        while (r2 == i || (r2 <= populationSize && r2 == r1)) && guard < 100
            r2 = randi(size(combined,1));
            guard = guard + 1;
        end
        mutant = population(i,:) + F .* (population(pbest,:) - population(i,:)) + ...
            F .* (population(r1,:) - combined(r2,:));
        mutant = min(upperBound, max(lowerBound, mutant));
        crossoverMask = rand(1, dimension) <= CR;
        crossoverMask(randi(dimension)) = true;
        trial(i,:) = population(i,:);
        trial(i,crossoverMask) = mutant(crossoverMask);
        trial(i,:) = repairFcn(trial(i,:));
    end

    trialScores = evaluate_population(trial, objectiveFcn, options.useParallel);
    evaluations = evaluations + populationSize;
    success = trialScores > scores;
    improvements = trialScores(success) - scores(success);
    if any(success)
        archive = [archive; population(success,:)]; %#ok<AGROW>
        maximumArchive = max(1, round(options.archiveRate * populationSize));
        if size(archive,1) > maximumArchive
            archive = archive(randperm(size(archive,1), maximumArchive),:);
        end
        population(success,:) = trial(success,:);
        scores(success) = trialScores(success);
        weights = improvements ./ sum(improvements);
        successfulF = usedF(success);
        successfulCR = usedCR(success);
        memoryF(memoryIndex) = sum(weights .* successfulF.^2) / ...
            max(eps, sum(weights .* successfulF));
        memoryCR(memoryIndex) = sum(weights .* successfulCR);
        memoryIndex = mod(memoryIndex, options.memorySize) + 1;
    end

    targetPopulationSize = round(initialPopulationSize - ...
        (initialPopulationSize - options.minPopulationSize) * iteration / options.maxIterations);
    targetPopulationSize = max(options.minPopulationSize, targetPopulationSize);
    if size(population,1) > targetPopulationSize
        [~, order] = sort(scores, 'descend');
        keep = order(1:targetPopulationSize);
        population = population(keep,:);
        scores = scores(keep);
    end

    [iterationBest, idx] = max(scores);
    if iterationBest > bestScore + options.functionTolerance
        bestScore = iterationBest;
        bestX = population(idx,:);
        stallCount = 0;
    else
        stallCount = stallCount + 1;
    end
    diversity = mean(std((population - lowerBound) ./ (upperBound-lowerBound), 0, 1));
    historyData(iteration,:) = [iteration, bestScore, mean(scores(isfinite(scores))), ...
        diversity, evaluations, size(population,1)];
    if options.debug
        fprintf('[DE] 迭代 %d/%d，当前最优 %.6f，种群 %d，评价次数 %d。\n', ...
            iteration, options.maxIterations, bestScore, size(population,1), evaluations);
    end
    if isfield(options,'liveCallback') && ~isempty(options.liveCallback)
        if isfield(options,'algorithmLabel')
            label = options.algorithmLabel;
        else
            label = 'DE';
        end
        state = struct('algorithm',label,'iteration',iteration,'bestScore',bestScore, ...
            'meanScore',mean(scores(isfinite(scores))),'diversity',diversity, ...
            'evaluations',evaluations,'populationSize',size(population,1));
        try
            options.liveCallback(state);
        catch ME
            warning('SmokePlot:LiveCallbackFailed','实时收敛图更新失败：%s',ME.message);
        end
    end
    if stallCount >= options.maxStallIterations
        historyData = historyData(1:iteration,:);
        if options.debug
            fprintf('[DE] 连续 %d 代无显著改善，提前停止。\n', stallCount);
        end
        break;
    end
end

history = array2table(historyData, 'VariableNames', ...
    {'Iteration','BestScore','MeanScore','Diversity','Evaluations','PopulationSize'});
[candidateScores, order] = sort(scores, 'descend');
candidates.x = population(order,:);
candidates.score = candidateScores;
end
