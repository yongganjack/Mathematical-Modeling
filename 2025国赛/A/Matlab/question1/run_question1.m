function output = run_question1(cfg)
%RUN_QUESTION1 计算题设固定 FY1 单弹策略对 M1 的有效遮蔽时长。

fprintf('\n========== 开始求解问题1（%s 配置）==========\n', cfg.name);
rng(cfg.masterSeed, 'twister');
problem = load_problem_data();
runDir = create_run_directory(cfg, 'question1');

plan = struct('uavIdx',1,'bombId',1,'targetId',1,'theta',pi,'speed',120, ...
    'releaseTime',1.5,'delay',3.6);
try
    fastEvaluation = evaluate_plan(plan, problem, cfg, 1, 'fast');
    verifyEvaluation = evaluate_plan(plan, problem, cfg, 1, 'verify');
    convergence = table(["fast";"verify"], ...
        [fastEvaluation.missiles.duration; verifyEvaluation.missiles.duration], ...
        [fastEvaluation.missiles.sampleCount; verifyEvaluation.missiles.sampleCount], ...
        [fastEvaluation.missiles.timeStep; verifyEvaluation.missiles.timeStep], ...
        'VariableNames', {'Level','Duration','SampleCount','TimeStep'});
    save_run_bundle(runDir, 'question1', cfg, problem, plan, verifyEvaluation, convergence);
    writetable(convergence, fullfile(runDir,'convergence.csv'));
    if cfg.enablePlots && cfg.showStaticFigures
        plot_plan_results(problem, plan, verifyEvaluation, runDir, 'Question 1', cfg);
    end
    if cfg.enablePlots && cfg.livePlots
        animate_plan_3d(problem,plan,verifyEvaluation,cfg,'Question 1');
    end
    print_plan_console_summary(problem,plan,verifyEvaluation,'问题1');
    fprintf('[问题1] 快速/高精度时长差：%.6g s。\n', ...
        abs(fastEvaluation.missiles.duration-verifyEvaluation.missiles.duration));
    fprintf('[问题1] 结果目录：%s\n', runDir);
catch ME
    fprintf(2, '[问题1异常] 求解失败：%s\n', ME.message);
    rethrow(ME);
end
output = struct('runDir',runDir,'plan',plan,'evaluation',verifyEvaluation,'convergence',convergence);
end
