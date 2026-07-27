function plan = decode_mixed_route(x,uavIdx,targetIds,problem)
%DECODE_MIXED_ROUTE 解码共享航向速度、逐弹时序和逐弹目标标签。

targetIds = targetIds(:).';
x = repair_mixed_route_decision(x,uavIdx,targetIds,problem);
bombCount = numel(targetIds);
releaseIdx = 3:(2+bombCount);
delayIdx = (3+bombCount):(2+2*bombCount);
template = struct('uavIdx',uavIdx,'bombId',0,'targetId',0,'theta',x(1), ...
    'speed',x(2),'releaseTime',0,'delay',0);
plan = repmat(template,1,bombCount);
for k = 1:bombCount
    plan(k).bombId = k;
    plan(k).targetId = targetIds(k);
    plan(k).releaseTime = x(releaseIdx(k));
    plan(k).delay = x(delayIdx(k));
end
end
