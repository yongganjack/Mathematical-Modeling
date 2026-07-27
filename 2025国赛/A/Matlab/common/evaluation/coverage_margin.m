function [margin, coverageRatio, detail] = coverage_margin(t, missileIdx, bombs, problem, targetSample, tolerance)
%COVERAGE_MARGIN 计算联合遮蔽裕量：最不利目标点的最近云团距离减半径。

detail = struct('activeBombs', [], 'visiblePointCount', 0, 'worstPoint', [NaN,NaN,NaN]);
active = find(arrayfun(@(b) t >= b.burstTime - tolerance && ...
    t <= b.cloudEndTime + tolerance, bombs));
if isempty(active)
    margin = inf;
    coverageRatio = 0;
    return;
end

mPos = missile_position(problem, missileIdx, t);
points = visible_target_points(targetSample, mPos, tolerance);
lineVectors = points - mPos;
lineNorm2 = sum(lineVectors.^2, 2);
minDistance = inf(size(points, 1), 1);

for idx = active
    cloudCenter = bombs(idx).burstPoint - ...
        problem.smokeSinkSpeed .* (t - bombs(idx).burstTime) .* [0,0,1];
    relative = cloudCenter - mPos;
    lambda = (lineVectors * relative.') ./ lineNorm2;
    lambda = min(1, max(0, lambda));
    closest = mPos + lambda .* lineVectors;
    distances = vecnorm(cloudCenter - closest, 2, 2);
    minDistance = min(minDistance, distances);
end

[worstDistance, worstIdx] = max(minDistance);
margin = worstDistance - problem.smokeRadius;
coverageRatio = mean(minDistance <= problem.smokeRadius + tolerance);
detail.activeBombs = active;
detail.visiblePointCount = size(points, 1);
detail.worstPoint = points(worstIdx, :);
end
