function plan = decode_single_bomb(x, uavIdx, targetId, problem, bombId)
%DECODE_SINGLE_BOMB 将4维变量解码为一枚烟幕弹计划。

if nargin < 5
    bombId = 1;
end
x = repair_single_bomb_decision(x, uavIdx, targetId, problem);
plan = struct('uavIdx', uavIdx, 'bombId', bombId, 'targetId', targetId, ...
    'theta', x(1), 'speed', x(2), 'releaseTime', x(3), 'delay', x(4));
end
