function x = repair_shared_route_decision(x, uavIdx, targetId, bombCount, problem)
%REPAIR_SHARED_ROUTE_DECISION 单目标路线修复的兼容包装。
x = repair_mixed_route_decision(x,uavIdx,repmat(targetId,1,bombCount),problem);
end
