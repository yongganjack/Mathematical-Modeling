function [bestX, bestScore, history, candidates] = improved_pso(objectiveFcn, lowerBound, upperBound, options, repairFcn)
%IMPROVED_PSO 自适应惯性、速度限幅和停滞部分重启的粒子群算法。

if nargin < 5 || isempty(repairFcn)
    repairFcn = @(x) min(upperBound, max(lowerBound, x));
end
dimension = numel(lowerBound);
if numel(upperBound) ~= dimension || any(upperBound <= lowerBound)
    error('SmokeOptimization:InvalidPSOBounds', 'PSO 上下界维度不一致或存在非正区间。');
end
rng(options.seed, 'twister');

swarmSize = options.swarmSize;
positions = stratified_population(swarmSize, lowerBound, upperBound);
if isfield(options,'initialPoints') && ~isempty(options.initialPoints)
    initialPoints = options.initialPoints;
    if size(initialPoints,2) ~= dimension
        error('SmokeOptimization:InvalidInitialPoints', 'PSO 几何初始点维度与决策变量不一致。');
    end
    initialCount = min(swarmSize,size(initialPoints,1));
    positions(1:initialCount,:) = initialPoints(1:initialCount,:);
end
for i = 1:swarmSize
    positions(i,:) = repairFcn(positions(i,:));
end
span = upperBound - lowerBound;
velocityLimit = options.velocityFraction .* span;
velocities = (2*rand(swarmSize, dimension)-1) .* velocityLimit;
scores = evaluate_population(positions, objectiveFcn, options.useParallel);
evaluations = swarmSize;

personalBest = positions;
personalScore = scores;
[bestScore, idxBest] = max(scores);
bestX = positions(idxBest, :);
stallCount = 0;
restartCount = 0;
historyData = zeros(options.maxIterations, 6);

for iteration = 1:options.maxIterations
    progress = (iteration - 1) / max(1, options.maxIterations - 1);
    inertia = options.inertiaMax - (options.inertiaMax - options.inertiaMin) * progress;
    diversity = mean(std((positions - lowerBound) ./ span, 0, 1));
    if diversity < 0.03
        inertia = min(options.inertiaMax, inertia + 0.12);
    end

    r1 = rand(swarmSize, dimension);
    r2 = rand(swarmSize, dimension);
    velocities = inertia .* velocities + ...
        options.cognitiveWeight .* r1 .* (personalBest - positions) + ...
        options.socialWeight .* r2 .* (bestX - positions);
    velocities = min(velocityLimit, max(-velocityLimit, velocities));
    positions = positions + velocities;
    outsideLow = positions < lowerBound;
    outsideHigh = positions > upperBound;
    velocities(outsideLow | outsideHigh) = -0.5 .* velocities(outsideLow | outsideHigh);
    positions = min(upperBound, max(lowerBound, positions));
    for i = 1:swarmSize
        positions(i,:) = repairFcn(positions(i,:));
    end

    scores = evaluate_population(positions, objectiveFcn, options.useParallel);
    evaluations = evaluations + swarmSize;
    improved = scores > personalScore;
    personalBest(improved,:) = positions(improved,:);
    personalScore(improved) = scores(improved);
    [iterationBest, idxBest] = max(scores);
    if iterationBest > bestScore + options.functionTolerance
        bestScore = iterationBest;
        bestX = positions(idxBest,:);
        stallCount = 0;
    else
        stallCount = stallCount + 1;
    end

    if stallCount >= options.maxStallIterations && iteration < options.maxIterations
        restartNumber = max(1, round(options.restartFraction * swarmSize));
        [~, order] = sort(personalScore, 'ascend');
        restartIdx = order(1:restartNumber);
        replacement = stratified_population(restartNumber, lowerBound, upperBound);
        for r = 1:restartNumber
            replacement(r,:) = repairFcn(replacement(r,:));
        end
        positions(restartIdx,:) = replacement;
        velocities(restartIdx,:) = 0;
        scores(restartIdx) = evaluate_population(replacement, objectiveFcn, options.useParallel);
        evaluations = evaluations + restartNumber;
        personalBest(restartIdx,:) = replacement;
        personalScore(restartIdx) = scores(restartIdx);
        [restartBest, localIdx] = max(scores(restartIdx));
        if restartBest > bestScore
            bestScore = restartBest;
            bestX = replacement(localIdx,:);
        end
        stallCount = 0;
        restartCount = restartCount + 1;
    end

    diversity = mean(std((positions - lowerBound) ./ span, 0, 1));
    historyData(iteration,:) = [iteration, bestScore, mean(scores(isfinite(scores))), ...
        diversity, evaluations, restartCount];
    if options.debug
        fprintf('[PSO] 迭代 %d/%d，当前最优 %.6f，群体离散度 %.4f，评价次数 %d。\n', ...
            iteration, options.maxIterations, bestScore, diversity, evaluations);
    end
    if isfield(options,'liveCallback') && ~isempty(options.liveCallback)
        if isfield(options,'algorithmLabel')
            label = options.algorithmLabel;
        else
            label = 'PSO';
        end
        state = struct('algorithm',label,'iteration',iteration,'bestScore',bestScore, ...
            'meanScore',mean(scores(isfinite(scores))),'diversity',diversity, ...
            'evaluations',evaluations,'populationSize',swarmSize);
        try
            options.liveCallback(state);
        catch ME
            warning('SmokePlot:LiveCallbackFailed','实时收敛图更新失败：%s',ME.message);
        end
    end
end

history = array2table(historyData, 'VariableNames', ...
    {'Iteration','BestScore','MeanScore','Diversity','Evaluations','Restarts'});
[candidateScores, order] = sort(personalScore, 'descend');
candidates.x = personalBest(order,:);
candidates.score = candidateScores;
end
