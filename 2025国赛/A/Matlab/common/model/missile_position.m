function position = missile_position(problem, missileIdx, t)
%MISSILE_POSITION 计算给定时刻的导弹位置。

if missileIdx < 1 || missileIdx > size(problem.missileInit, 1)
    error('SmokeModel:MissileIndexOutOfRange', '导弹索引 %d 不在有效范围内。', missileIdx);
end
position = problem.missileInit(missileIdx, :) + ...
    problem.missileSpeed .* t .* problem.missileDirection(missileIdx, :);
end
