function x = repair_single_bomb_decision(x, uavIdx, targetId, problem)
%REPAIR_SINGLE_BOMB_DECISION 修复单弹变量到物理可行域。

x(1) = mod(x(1) + pi, 2*pi) - pi;
x(2) = min(problem.uavSpeedBounds(2), max(problem.uavSpeedBounds(1), x(2)));
x(4) = min(problem.maxDelay(uavIdx), max(0, x(4)));
latestRelease = max(0, problem.hitTime(targetId) - x(4));
x(3) = min(latestRelease, max(0, x(3)));
end
