function result = evaluate_plan(plan, problem, cfg, missileIds, mode)
%EVALUATE_PLAN 对一个完整投放计划计算连续遮蔽区间和诊断指标。

if nargin < 4 || isempty(missileIds)
    missileIds = 1:numel(problem.hitTime);
end
if nargin < 5
    mode = 'verify';
end

[isFeasible, violations] = validate_plan(plan, problem);
result = struct('feasible', isFeasible, 'violations', violations, 'bombs', struct([]), ...
    'missiles', struct([]), 'totalDuration', 0, 'minimumDuration', 0, ...
    'auxiliaryScore', -inf, 'mode', mode);
if ~isFeasible
    result.auxiliaryScore = -1e6 - sum(struct2array(violations));
    return;
end

bombs = derive_bombs(plan, problem);
result.bombs = bombs;
if strcmpi(mode, 'fast')
    sampleCfg = cfg.fastSample;
    timeStep = cfg.fastTimeStep;
else
    sampleCfg = cfg.verifySample;
    timeStep = cfg.verifyTimeStep;
end
targetSample = create_target_samples(problem.target, sampleCfg);

missileTemplate = struct('missileIdx', 0, 'duration', 0, 'intervals', zeros(0,2), ...
    'coverageIntegral', 0, 'timeGrid', [], 'margin', [], 'coverageRatio', [], ...
    'boundaryResidual', NaN, 'sampleCount', targetSample.count, 'timeStep', timeStep);
missileResults = repmat(missileTemplate, 1, numel(missileIds));

for jj = 1:numel(missileIds)
    missileIdx = missileIds(jj);
    tEnd = problem.hitTime(missileIdx);
    eventTimes = [0, tEnd];
    if ~isempty(bombs)
        eventTimes = [eventTimes, [bombs.burstTime], [bombs.cloudEndTime]]; %#ok<AGROW>
    end
    eventTimes = eventTimes(isfinite(eventTimes) & eventTimes >= 0 & eventTimes <= tEnd);
    regularGrid = 0:timeStep:tEnd;
    if isempty(regularGrid) || regularGrid(end) < tEnd
        regularGrid(end+1) = tEnd;
    end
    timeGrid = unique([regularGrid, eventTimes]);
    margins = inf(size(timeGrid));
    coverage = zeros(size(timeGrid));
    for tt = 1:numel(timeGrid)
        [margins(tt), coverage(tt)] = coverage_margin(timeGrid(tt), missileIdx, bombs, ...
            problem, targetSample, cfg.geometryTolerance);
    end
    states = margins <= 0;
    intervals = zeros(0, 2);
    currentStart = NaN;
    if states(1)
        currentStart = timeGrid(1);
    end
    for tt = 1:(numel(timeGrid)-1)
        if states(tt) ~= states(tt+1)
            boundary = refine_transition(timeGrid(tt), timeGrid(tt+1), states(tt), ...
                missileIdx, bombs, problem, targetSample, cfg);
            if ~states(tt) && states(tt+1)
                currentStart = boundary;
            else
                if isnan(currentStart)
                    currentStart = timeGrid(tt);
                end
                intervals(end+1, :) = [currentStart, boundary]; %#ok<AGROW>
                currentStart = NaN;
            end
        end
    end
    if states(end)
        if isnan(currentStart)
            currentStart = timeGrid(end);
        end
        intervals(end+1, :) = [currentStart, timeGrid(end)]; %#ok<AGROW>
    end
    intervals = merge_intervals_local(intervals, cfg.intervalMergeTolerance);
    duration = sum(max(0, intervals(:,2) - intervals(:,1)));

    residuals = [];
    for rr = 1:size(intervals,1)
        endpoints = intervals(rr, :);
        for ee = 1:2
            endpoint = endpoints(ee);
            if min(abs(eventTimes - endpoint)) > 10*cfg.rootTolerance
                value = coverage_margin(endpoint, missileIdx, bombs, problem, targetSample, cfg.geometryTolerance);
                if isfinite(value)
                    residuals(end+1) = abs(value); %#ok<AGROW>
                end
            end
        end
    end
    if isempty(residuals)
        boundaryResidual = NaN;
    else
        boundaryResidual = max(residuals);
    end

    missileResults(jj) = missileTemplate;
    missileResults(jj).missileIdx = missileIdx;
    missileResults(jj).duration = duration;
    missileResults(jj).intervals = intervals;
    missileResults(jj).coverageIntegral = trapz(timeGrid, coverage);
    missileResults(jj).timeGrid = timeGrid;
    missileResults(jj).margin = margins;
    missileResults(jj).coverageRatio = coverage;
    missileResults(jj).boundaryResidual = boundaryResidual;
end

result.missiles = missileResults;
durations = [missileResults.duration];
result.totalDuration = sum(durations);
result.minimumDuration = min(durations);
finiteBestMargins = zeros(1,numel(missileResults));
for j = 1:numel(missileResults)
    finiteMargins = missileResults(j).margin(isfinite(missileResults(j).margin));
    if isempty(finiteMargins)
        finiteBestMargins(j) = 1e6;
    else
        finiteBestMargins(j) = max(0,min(finiteMargins));
    end
end
result.auxiliaryScore = result.totalDuration + cfg.coverageWeight * ...
    sum([missileResults.coverageIntegral]) - cfg.proximityWeight * sum(finiteBestMargins);
end

function merged = merge_intervals_local(intervals, tolerance)
if isempty(intervals)
    merged = intervals;
    return;
end
intervals = sortrows(intervals, 1);
merged = intervals(1, :);
for k = 2:size(intervals, 1)
    if intervals(k,1) <= merged(end,2) + tolerance
        merged(end,2) = max(merged(end,2), intervals(k,2));
    else
        merged(end+1,:) = intervals(k,:); %#ok<AGROW>
    end
end
end
