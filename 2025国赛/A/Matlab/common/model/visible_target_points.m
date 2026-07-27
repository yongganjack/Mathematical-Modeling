function points = visible_target_points(sample, missilePos, tolerance)
%VISIBLE_TARGET_POINTS 按凸圆柱表面外法向筛选当前视角的可见点。

viewVectors = missilePos - sample.points;
visibleMask = sum(sample.normals .* viewVectors, 2) >= -tolerance;
points = sample.points(visibleMask, :);
if isempty(points)
    error('SmokeModel:EmptyVisiblePoints', '当前导弹视角未筛选到任何目标可见采样点。');
end
end
