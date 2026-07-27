function [isFeasible, violations] = validate_plan(plan, problem)
%VALIDATE_PLAN 检查空值、边界、同机共享航线、投放间隔和起爆高度。

violations = struct('nonfinite', 0, 'index', 0, 'speed', 0, 'time', 0, ...
    'delay', 0, 'burstHeight', 0, 'releaseInterval', 0, 'sharedRoute', 0);
if isempty(plan)
    isFeasible = true;
    return;
end

for k = 1:numel(plan)
    p = plan(k);
    numericValues = [p.uavIdx, p.theta, p.speed, p.releaseTime, p.delay];
    if any(~isfinite(numericValues))
        violations.nonfinite = violations.nonfinite + 1;
        continue;
    end
    if p.uavIdx < 1 || p.uavIdx > size(problem.uavInit, 1) || p.uavIdx ~= round(p.uavIdx)
        violations.index = violations.index + 1;
        continue;
    end
    violations.speed = violations.speed + max(0, problem.uavSpeedBounds(1) - p.speed) + ...
        max(0, p.speed - problem.uavSpeedBounds(2));
    violations.time = violations.time + max(0, -p.releaseTime);
    violations.delay = violations.delay + max(0, -p.delay) + ...
        max(0, p.delay - problem.maxDelay(p.uavIdx));
    burstHeight = problem.uavInit(p.uavIdx, 3) - 0.5*problem.gravity*p.delay^2;
    violations.burstHeight = violations.burstHeight + max(0, -burstHeight);
    if isfield(p, 'targetId') && p.targetId >= 1 && p.targetId <= numel(problem.hitTime)
        endTime = problem.hitTime(p.targetId);
    else
        endTime = max(problem.hitTime);
    end
    violations.time = violations.time + max(0, p.releaseTime + p.delay - endTime);
end

for u = 1:size(problem.uavInit, 1)
    idx = find([plan.uavIdx] == u);
    if isempty(idx)
        continue;
    end
    thetaValues = [plan(idx).theta];
    speedValues = [plan(idx).speed];
    violations.sharedRoute = violations.sharedRoute + ...
        max(0, max(abs(wrapToPiLocal(thetaValues - thetaValues(1)))) - 1e-8) + ...
        max(0, max(abs(speedValues - speedValues(1))) - 1e-8);
    releaseTimes = sort([plan(idx).releaseTime]);
    if numel(releaseTimes) > 1
        violations.releaseInterval = violations.releaseInterval + ...
            sum(max(0, problem.minReleaseInterval - diff(releaseTimes)));
    end
end

values = struct2array(violations);
isFeasible = all(values <= 1e-8);
end

function angle = wrapToPiLocal(angle)
angle = mod(angle + pi, 2*pi) - pi;
end
