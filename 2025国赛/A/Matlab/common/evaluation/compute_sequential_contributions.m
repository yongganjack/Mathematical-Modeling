function contributions = compute_sequential_contributions(plan, problem, cfg)
%COMPUTE_SEQUENTIAL_CONTRIBUTIONS 按全局投放顺序计算每枚弹对标记导弹的新增时长。

if isempty(plan)
    contributions = [];
    return;
end
[~, order] = sortrows([[plan.releaseTime].', [plan.uavIdx].', [plan.bombId].'], [1,2,3]);
orderedPlan = plan(order);
contributionsOrdered = zeros(1, numel(plan));
previousDurations = zeros(1, numel(problem.hitTime));
for k = 1:numel(orderedPlan)
    prefix = orderedPlan(1:k);
    current = evaluate_plan(prefix, problem, cfg, 1:numel(problem.hitTime), 'verify');
    currentDurations = zeros(1, numel(problem.hitTime));
    for j = 1:numel(current.missiles)
        currentDurations(current.missiles(j).missileIdx) = current.missiles(j).duration;
    end
    targetId = orderedPlan(k).targetId;
    contributionsOrdered(k) = max(0, currentDurations(targetId) - previousDurations(targetId));
    previousDurations = currentDurations;
end
contributions = zeros(1, numel(plan));
contributions(order) = contributionsOrdered;
end
