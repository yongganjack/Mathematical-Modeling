function bombs = derive_bombs(plan, problem)
%DERIVE_BOMBS 由独立决策变量正向计算投放点、起爆点与有效时间窗。

if isempty(plan)
    bombs = struct([]);
    return;
end

template = struct('uavIdx', 0, 'bombId', 0, 'targetId', 0, 'theta', 0, ...
    'speed', 0, 'releaseTime', 0, 'delay', 0, 'burstTime', 0, ...
    'direction', zeros(1,3), 'releasePoint', zeros(1,3), ...
    'burstPoint', zeros(1,3), 'cloudEndTime', 0);
bombs = repmat(template, 1, numel(plan));

for k = 1:numel(plan)
    p = plan(k);
    u0 = problem.uavInit(p.uavIdx, :);
    direction = [cos(p.theta), sin(p.theta), 0];
    burstTime = p.releaseTime + p.delay;
    releasePoint = u0 + p.speed .* p.releaseTime .* direction;
    burstPoint = u0 + p.speed .* burstTime .* direction - ...
        0.5 .* problem.gravity .* p.delay.^2 .* [0, 0, 1];
    if problem.smokeSinkSpeed > 0
        groundEnd = burstTime + (burstPoint(3) + problem.smokeRadius) / problem.smokeSinkSpeed;
    else
        groundEnd = inf;
    end

    bombs(k) = template;
    fields = fieldnames(template);
    for f = 1:numel(fields)
        if isfield(p, fields{f})
            bombs(k).(fields{f}) = p.(fields{f});
        end
    end
    bombs(k).uavIdx = p.uavIdx;
    bombs(k).bombId = p.bombId;
    if isfield(p, 'targetId')
        bombs(k).targetId = p.targetId;
    else
        bombs(k).targetId = 1;
    end
    bombs(k).theta = p.theta;
    bombs(k).speed = p.speed;
    bombs(k).releaseTime = p.releaseTime;
    bombs(k).delay = p.delay;
    bombs(k).burstTime = burstTime;
    bombs(k).direction = direction;
    bombs(k).releasePoint = releasePoint;
    bombs(k).burstPoint = burstPoint;
    bombs(k).cloudEndTime = min(burstTime + problem.smokeLifetime, groundEnd);
end
end
