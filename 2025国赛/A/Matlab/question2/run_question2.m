function output = run_question2(cfg)
%RUN_QUESTION2 使用改进 PSO 与自适应 DE 优化 FY1 单弹策略。

fprintf('\n========== 开始求解问题2（%s 配置）==========\n', cfg.name);
problem = load_problem_data();
runDir = create_run_directory(cfg, 'question2');
lowerBound = [-pi, problem.uavSpeedBounds(1), 0, 0];
upperBound = [ pi, problem.uavSpeedBounds(2), problem.hitTime(1), problem.maxDelay(1)];
repair = @(x) repair_single_bomb_decision(x,1,1,problem);
decoder = @(x) decode_single_bomb(x,1,1,problem,1);
objective = @(x) single_objective(x,decoder,problem,cfg,1);
[psoOptions,deOptions] = build_solver_options(cfg,20);
if cfg.enablePlots && cfg.livePlots
    liveMonitor = create_live_convergence_monitor('Question 2');
    psoOptions.liveCallback = liveMonitor.callback;
    psoOptions.algorithmLabel = 'PSO';
    deOptions.liveCallback = liveMonitor.callback;
    deOptions.algorithmLabel = 'DE';
end

try
    geometrySeeds = generate_geometry_seeds(problem,1,1);
    if isempty(geometrySeeds)
        warning('SmokeOptimization:NoGeometrySeeds','未生成几何初始候选，将退回分层随机初始化。');
    else
        seedScores = evaluate_population(geometrySeeds,objective,false);
        [seedScores,seedOrder] = sort(seedScores,'descend');
        geometrySeeds = geometrySeeds(seedOrder,:);
        keepSeeds = min(max(psoOptions.swarmSize,deOptions.populationSize),size(geometrySeeds,1));
        geometrySeeds = geometrySeeds(1:keepSeeds,:);
        psoOptions.initialPoints = geometrySeeds;
        deOptions.initialPoints = geometrySeeds;
        seedPlan = decoder(geometrySeeds(1,:));
        seedEvaluation = evaluate_plan(seedPlan,problem,cfg,1,'fast');
        fprintf('[问题2] 已生成 %d 个几何初始候选，最佳初始真实时长 %.6f s，搜索评分 %.6f。\n', ...
            size(geometrySeeds,1),seedEvaluation.totalDuration,seedScores(1));
    end
    [psoX,~,psoHistory,psoCandidates] = improved_pso(objective,lowerBound,upperBound,psoOptions,repair);
    [deX,~,deHistory,deCandidates] = adaptive_de(objective,lowerBound,upperBound,deOptions,repair);
    pool = [psoX; deX; psoCandidates.x(1:min(cfg.q2.keepCandidates,size(psoCandidates.x,1)),:); ...
        deCandidates.x(1:min(cfg.q2.keepCandidates,size(deCandidates.x,1)),:)];
    [plan,evaluation,verification] = verify_candidate_set(pool,decoder,problem,cfg,1,2*cfg.q2.keepCandidates+2);
    history.pso = psoHistory;
    history.de = deHistory;
    history.verification = verification;
    save_run_bundle(runDir,'question2',cfg,problem,plan,evaluation,history);
    writetable(verification,fullfile(runDir,'candidate_verification.csv'));
    if cfg.enablePlots && cfg.showStaticFigures
        plot_plan_results(problem,plan,evaluation,runDir,'Question 2',cfg);
    end
    if cfg.enablePlots && cfg.livePlots
        animate_plan_3d(problem,plan,evaluation,cfg,'Question 2');
    end
    print_plan_console_summary(problem,plan,evaluation,'问题2');
    fprintf('[问题2] 结果目录：%s\n',runDir);
catch ME
    fprintf(2,'[问题2异常] 求解失败：%s\n',ME.message);
    rethrow(ME);
end
output = struct('runDir',runDir,'plan',plan,'evaluation',evaluation,'history',history);
end

function score = single_objective(x,decoder,problem,cfg,missileIdx)
evaluation = evaluate_plan(decoder(x),problem,cfg,missileIdx,'fast');
score = evaluation.auxiliaryScore;
end
