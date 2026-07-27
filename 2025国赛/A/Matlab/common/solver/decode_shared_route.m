function plan = decode_shared_route(x, uavIdx, targetId, bombCount, problem)
%DECODE_SHARED_ROUTE 单目标共享路线解码的兼容包装。
plan = decode_mixed_route(x,uavIdx,repmat(targetId,1,bombCount),problem);
end
