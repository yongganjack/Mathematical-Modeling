function problem = load_problem_data()
%LOAD_PROBLEM_DATA 按赛题原文构造统一输入结构，并执行基础校验。

problem.missileIds = ["M1", "M2", "M3"];
problem.missileInit = [20000, 0, 2000; 19000, 600, 2100; 18000, -600, 1900];
problem.uavIds = ["FY1", "FY2", "FY3", "FY4", "FY5"];
problem.uavInit = [17800, 0, 1800; 12000, 1400, 1400; 6000, -3000, 700; ...
    11000, 2000, 1800; 13000, -2000, 1300];

problem.target.center = [0, 200, 0];
problem.target.radius = 7;
problem.target.height = 10;
problem.missileSpeed = 300;
problem.uavSpeedBounds = [70, 140];
problem.smokeRadius = 10;
problem.smokeSinkSpeed = 3;
problem.smokeLifetime = 20;
problem.minReleaseInterval = 1;
problem.gravity = 9.8;

if ~isequal(size(problem.missileInit), [3, 3])
    error('SmokeModel:MissileDimension', '导弹初始位置必须是 3×3 数组。');
end
if ~isequal(size(problem.uavInit), [5, 3])
    error('SmokeModel:UavDimension', '无人机初始位置必须是 5×3 数组。');
end
if any(~isfinite(problem.missileInit), 'all') || any(~isfinite(problem.uavInit), 'all')
    error('SmokeModel:InvalidCoordinate', '导弹或无人机坐标包含 NaN/Inf。');
end
if any(vecnorm(problem.missileInit, 2, 2) <= 0)
    error('SmokeModel:InvalidMissilePosition', '导弹初始位置不能位于坐标原点。');
end
if problem.uavSpeedBounds(1) > problem.uavSpeedBounds(2) || ...
        any([problem.smokeRadius, problem.smokeLifetime, problem.gravity] <= 0)
    error('SmokeModel:InvalidPhysicalParameter', '速度边界或烟幕物理参数不满足要求。');
end

problem.missileDirection = -problem.missileInit ./ vecnorm(problem.missileInit, 2, 2);
problem.hitTime = vecnorm(problem.missileInit, 2, 2) ./ problem.missileSpeed;
problem.maxDelay = sqrt(2 .* problem.uavInit(:, 3) ./ problem.gravity);
end
