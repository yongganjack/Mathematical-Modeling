function print_plan_console_summary(problem,plan,evaluation,questionName)
%PRINT_PLAN_CONSOLE_SUMMARY 在控制台输出Q1-Q5统一策略明细。

fprintf('\n========== %s 策略明细（高精度回代）==========\n',questionName);
if isempty(plan)
    fprintf('当前方案没有实际投放烟幕弹。\n');
else
    bombs = derive_bombs(plan,problem);
    [~,order] = sortrows([[bombs.releaseTime].',[bombs.uavIdx].',[bombs.bombId].'],[1,2,3]);
    for ii = 1:numel(order)
        b = bombs(order(ii));
        fprintf('无人机 %s | 烟幕弹 %d | 主要目标 M%d\n', ...
            problem.uavIds(b.uavIdx),b.bombId,b.targetId);
        fprintf('  飞行方向：航向角 %.9f deg，单位向量 (%.9f, %.9f, %.9f)\n', ...
            mod(rad2deg(b.theta),360),b.direction);
        fprintf('  飞行速度：%.9f m/s\n',b.speed);
        fprintf('  投放时刻：%.9f s；投放点：(%.9f, %.9f, %.9f) m\n', ...
            b.releaseTime,b.releasePoint);
        fprintf('  起爆时刻：%.9f s；起爆延时：%.9f s；起爆点：(%.9f, %.9f, %.9f) m\n', ...
            b.burstTime,b.delay,b.burstPoint);
    end
end
fprintf('各导弹有效遮蔽时长：');
for j=1:numel(evaluation.missiles)
    fprintf(' M%d=%.9f s',evaluation.missiles(j).missileIdx,evaluation.missiles(j).duration);
end
fprintf('\n总遮蔽时长：%.9f s\n',evaluation.totalDuration);
fprintf('================================================\n');
end
