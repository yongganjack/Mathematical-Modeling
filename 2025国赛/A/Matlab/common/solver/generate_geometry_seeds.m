function seeds = generate_geometry_seeds(problem, uavIdx, missileIdx)
%GENERATE_GEOMETRY_SEEDS 由导弹—目标中心视线与无人机可达域反演单弹初始候选。

u0 = problem.uavInit(uavIdx,:);
targetPoint = problem.target.center + [0,0,problem.target.height/2];
tHit = problem.hitTime(missileIdx);
tBurstGrid = linspace(max(1,0.04*tHit),min(0.72*tHit,42),28);
lambdaGrid = unique([linspace(0.015,0.18,28),linspace(0.20,0.55,15)]);
seeds = zeros(0,4);

for tBurst = tBurstGrid
    missilePos = missile_position(problem,missileIdx,tBurst);
    for lambda = lambdaGrid
        desiredPoint = missilePos + lambda.*(targetPoint-missilePos);
        if desiredPoint(3) < 0 || desiredPoint(3) > u0(3)
            continue;
        end
        horizontal = desiredPoint(1:2)-u0(1:2);
        speed = norm(horizontal)/tBurst;
        if speed < problem.uavSpeedBounds(1) || speed > problem.uavSpeedBounds(2)
            continue;
        end
        delay = sqrt(max(0,2*(u0(3)-desiredPoint(3))/problem.gravity));
        releaseTime = tBurst-delay;
        if releaseTime < 0 || releaseTime+delay > tHit
            continue;
        end
        theta = atan2(horizontal(2),horizontal(1));
        seeds(end+1,:) = [theta,speed,releaseTime,delay]; %#ok<AGROW>
    end
end

% 加入问题1固定策略作为已知可行参考，但不把其结果当作问题2最优解。
if uavIdx == 1 && missileIdx == 1
    seeds(end+1,:) = [pi,120,1.5,3.6];
end
seeds = unique(round(seeds,10),'rows','stable');
end
