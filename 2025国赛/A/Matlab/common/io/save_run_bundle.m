function save_run_bundle(runDir, questionName, cfg, problem, plan, evaluation, history)
%SAVE_RUN_BUNDLE 保存完整精度解、摘要、区间、配置和优化历史。

save(fullfile(runDir, 'raw_solution.mat'), 'questionName', 'cfg', 'problem', ...
    'plan', 'evaluation', 'history', '-v7.3');

summary.question = questionName;
summary.configName = cfg.name;
summary.seed = cfg.masterSeed;
summary.feasible = evaluation.feasible;
summary.totalDuration = evaluation.totalDuration;
summary.minimumDuration = evaluation.minimumDuration;
summary.missileDurations = arrayfun(@(x) x.duration, evaluation.missiles);
summary.bombCount = numel(plan);
summary.note = '本文件记录实际 MATLAB 运行结果；是否为最终竞赛值取决于所用配置和收敛复核。';
write_json_file(fullfile(runDir, 'summary.json'), summary);
write_json_file(fullfile(runDir, 'config.json'), cfg);

missileColumn = strings(0,1);
startColumn = zeros(0,1);
endColumn = zeros(0,1);
durationColumn = zeros(0,1);
for j = 1:numel(evaluation.missiles)
    intervals = evaluation.missiles(j).intervals;
    n = size(intervals,1);
    missileColumn = [missileColumn; repmat(compose("M%d", evaluation.missiles(j).missileIdx), n, 1)]; %#ok<AGROW>
    startColumn = [startColumn; intervals(:,1)]; %#ok<AGROW>
    endColumn = [endColumn; intervals(:,2)]; %#ok<AGROW>
    durationColumn = [durationColumn; intervals(:,2)-intervals(:,1)]; %#ok<AGROW>
end
intervalTable = table(missileColumn, startColumn, endColumn, durationColumn, ...
    'VariableNames', {'Missile','StartTime','EndTime','Duration'});
writetable(intervalTable, fullfile(runDir, 'intervals.csv'));

if istable(history)
    writetable(history, fullfile(runDir, 'optimization_history.csv'));
elseif isstruct(history)
    names = fieldnames(history);
    for k = 1:numel(names)
        if istable(history.(names{k}))
            writetable(history.(names{k}), fullfile(runDir, ...
                ['optimization_history_', names{k}, '.csv']));
        end
    end
end
end
