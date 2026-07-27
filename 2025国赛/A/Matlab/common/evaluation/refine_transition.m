function boundary = refine_transition(tLeft, tRight, leftState, missileIdx, bombs, problem, sample, cfg)
%REFINE_TRANSITION 使用布尔二分法细化遮蔽状态变化边界。

for iter = 1:80
    if tRight - tLeft <= cfg.rootTolerance
        break;
    end
    mid = 0.5 * (tLeft + tRight);
    margin = coverage_margin(mid, missileIdx, bombs, problem, sample, cfg.geometryTolerance);
    midState = margin <= 0;
    if midState == leftState
        tLeft = mid;
    else
        tRight = mid;
    end
end
boundary = 0.5 * (tLeft + tRight);
end
