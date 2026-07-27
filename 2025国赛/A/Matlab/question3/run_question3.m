function output = run_question3(cfg)
%RUN_QUESTION3 优化 FY1 共享航线与三枚烟幕弹投放时序。

fprintf('\n========== 开始求解问题3（%s 配置）==========\n',cfg.name);
problem = load_problem_data();
runDir = create_run_directory(cfg,'question3');
bombCount = 3;
lowerBound = [-pi,70,zeros(1,bombCount),zeros(1,bombCount)];
upperBound = [ pi,140,repmat(problem.hitTime(1),1,bombCount), ...
    repmat(problem.maxDelay(1),1,bombCount)];
repair = @(x) repair_shared_route_decision(x,1,1,bombCount,problem);
decoder = @(x) decode_shared_route(x,1,1,bombCount,problem);
objective = @(x) multi_objective(x,decoder,problem,cfg,1);
[psoOptions,deOptions] = build_solver_options(cfg,30);
if cfg.enablePlots && cfg.livePlots
    liveMonitor = create_live_convergence_monitor('Question 3');
    psoOptions.liveCallback = liveMonitor.callback;
    psoOptions.algorithmLabel = 'PSO';
    deOptions.liveCallback = liveMonitor.callback;
    deOptions.algorithmLabel = 'DE';
end

try
    [psoX,~,psoHistory,psoCandidates] = improved_pso(objective,lowerBound,upperBound,psoOptions,repair);
    [deX,~,deHistory,deCandidates] = adaptive_de(objective,lowerBound,upperBound,deOptions,repair);
    pool = [psoX;deX;psoCandidates.x(1:min(cfg.q3.keepCandidates,size(psoCandidates.x,1)),:); ...
        deCandidates.x(1:min(cfg.q3.keepCandidates,size(deCandidates.x,1)),:)];
    [plan,evaluation,verification] = verify_candidate_set(pool,decoder,problem,cfg,1,2*cfg.q3.keepCandidates+2);
    history.pso = psoHistory;
    history.de = deHistory;
    history.verification = verification;
    save_run_bundle(runDir,'question3',cfg,problem,plan,evaluation,history);
    excelDir = fullfile(runDir,'excel'); mkdir(excelDir);
    template = fullfile(cfg.projectRoot,'00_赛题资料','附件','result1.xlsx');
    validation = export_result_workbook(3,template,fullfile(excelDir,'result1.xlsx'),plan,problem,cfg);
    write_json_file(fullfile(excelDir,'export_validation.json'),validation);
    if cfg.enablePlots && cfg.showStaticFigures
        plot_plan_results(problem,plan,evaluation,runDir,'Question 3',cfg);
    end
    if cfg.enablePlots && cfg.livePlots
        animate_plan_3d(problem,plan,evaluation,cfg,'Question 3');
    end
    print_plan_console_summary(problem,plan,evaluation,'问题3');
    fprintf('[问题3] 结果目录：%s\n',runDir);
catch ME
    fprintf(2,'[问题3异常] 求解失败：%s\n',ME.message);
    rethrow(ME);
end
output = struct('runDir',runDir,'plan',plan,'evaluation',evaluation,'history',history);
end

function score = multi_objective(x,decoder,problem,cfg,missileIdx)
evaluation = evaluate_plan(decoder(x),problem,cfg,missileIdx,'fast');
score = evaluation.auxiliaryScore;
end
