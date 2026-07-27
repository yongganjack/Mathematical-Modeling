function x = repair_mixed_route_decision(x,uavIdx,targetIds,problem)
%REPAIR_MIXED_ROUTE_DECISION 逐弹修复共享航向速度下的多目标时序。

targetIds = targetIds(:).';
bombCount = numel(targetIds);
if numel(x) ~= 2 + 2*bombCount
    error('SmokeOptimization:MixedRouteDimension','混合目标路线变量维度错误。');
end
x(1) = mod(x(1)+pi,2*pi)-pi;
x(2) = min(problem.uavSpeedBounds(2),max(problem.uavSpeedBounds(1),x(2)));
releaseIdx = 3:(2+bombCount);
delayIdx = (3+bombCount):(2+2*bombCount);

delays = min(problem.maxDelay(uavIdx),max(0,x(delayIdx)));
deadlines = problem.hitTime(targetIds)-delays;
deadlines = max(0,deadlines);

% 先从后向前形成满足相邻投放间隔的逐弹最晚时刻。
latest = deadlines;
for k = bombCount-1:-1:1
    latest(k) = min(latest(k),latest(k+1)-problem.minReleaseInterval);
end
if latest(1) < 0
    % 题面参数下通常不会触发；若触发则缩短延时，保留一个可行基线。
    delays = min(delays,max(0,problem.hitTime(targetIds)-((0:bombCount-1)*problem.minReleaseInterval)));
    deadlines = problem.hitTime(targetIds)-delays;
    latest = deadlines;
    for k = bombCount-1:-1:1
        latest(k) = min(latest(k),latest(k+1)-problem.minReleaseInterval);
    end
end

rawReleases = max(0,x(releaseIdx));
releases = zeros(1,bombCount);
releases(1) = min(rawReleases(1),max(0,latest(1)));
for k = 2:bombCount
    lower = releases(k-1)+problem.minReleaseInterval;
    releases(k) = min(max(rawReleases(k),lower),latest(k));
end
for k = 1:bombCount
    releases(k) = max(0,min(releases(k),deadlines(k)));
    delays(k) = min(delays(k),max(0,problem.hitTime(targetIds(k))-releases(k)));
end
x(releaseIdx) = releases;
x(delayIdx) = delays;
end
