function output = run_question4(cfg)
%RUN_QUESTION4 优化 FY1、FY2、FY3 各一枚烟幕弹的联合策略。

fprintf('\n========== 开始求解问题4（%s 配置）==========\n',cfg.name);
problem = load_problem_data();
runDir = create_run_directory(cfg,'question4');
lowerBound = zeros(1,12); upperBound = zeros(1,12);
for u = 1:3
    idx = (u-1)*4+(1:4);
    lowerBound(idx) = [-pi,70,0,0];
    upperBound(idx) = [pi,140,problem.hitTime(1),problem.maxDelay(u)];
end
repair = @(x) repair_q4(x,problem);
decoder = @(x) decode_q4(x,problem);
objective = @(x) q4_objective(x,decoder,problem,cfg);
[psoOptions,deOptions] = build_solver_options(cfg,40);
[initialPoints,initialTable] = generate_q4_initial_points(problem,cfg,objective);
psoOptions.initialPoints = initialPoints;
deOptions.initialPoints = initialPoints;
fprintf('[问题4] 已构造 %d 个三机几何组合初值，最佳快速评分 %.6f。\n', ...
    size(initialPoints,1),initialTable.Score(1));
if cfg.enablePlots && cfg.livePlots
    liveMonitor = create_live_convergence_monitor('Question 4');
    psoOptions.liveCallback = liveMonitor.callback;
    psoOptions.algorithmLabel = 'PSO';
    deOptions.liveCallback = liveMonitor.callback;
    deOptions.algorithmLabel = 'DE';
end

try
    [psoX,~,psoHistory,psoCandidates] = improved_pso(objective,lowerBound,upperBound,psoOptions,repair);
    [deX,~,deHistory,deCandidates] = adaptive_de(objective,lowerBound,upperBound,deOptions,repair);
    preliminaryPool = [psoX;deX;initialPoints; ...
        psoCandidates.x(1:min(cfg.q4.keepCandidates,size(psoCandidates.x,1)),:); ...
        deCandidates.x(1:min(cfg.q4.keepCandidates,size(deCandidates.x,1)),:)];
    preliminaryScores = evaluate_population(preliminaryPool,objective,false);
    [~,bestPreliminary] = max(preliminaryScores);
    [blockX,blockHistory] = block_refine_q4(preliminaryPool(bestPreliminary,:),problem,cfg,objective);
    pool = [blockX;psoX;deX;initialPoints;preliminaryPool];
    [plan,evaluation,verification] = verify_candidate_set(pool,decoder,problem,cfg,1,2*cfg.q4.keepCandidates+2);
    history.pso = psoHistory;
    history.de = deHistory;
    history.verification = verification;
    history.initial = initialTable;
    history.block = blockHistory;
    save_run_bundle(runDir,'question4',cfg,problem,plan,evaluation,history);
    excelDir = fullfile(runDir,'excel'); mkdir(excelDir);
    template = fullfile(cfg.projectRoot,'00_赛题资料','附件','result2.xlsx');
    validation = export_result_workbook(4,template,fullfile(excelDir,'result2.xlsx'),plan,problem,cfg);
    write_json_file(fullfile(excelDir,'export_validation.json'),validation);
    if cfg.enablePlots && cfg.showStaticFigures
        plot_plan_results(problem,plan,evaluation,runDir,'Question 4',cfg);
    end
    if cfg.enablePlots && cfg.livePlots
        animate_plan_3d(problem,plan,evaluation,cfg,'Question 4');
    end
    print_plan_console_summary(problem,plan,evaluation,'问题4');
    fprintf('[问题4] 结果目录：%s\n',runDir);
catch ME
    fprintf(2,'[问题4异常] 求解失败：%s\n',ME.message);
    rethrow(ME);
end
output = struct('runDir',runDir,'plan',plan,'evaluation',evaluation,'history',history);
end

function x = repair_q4(x,problem)
for u = 1:3
    idx = (u-1)*4+(1:4);
    x(idx) = repair_single_bomb_decision(x(idx),u,1,problem);
end
end

function plan = decode_q4(x,problem)
x = repair_q4(x,problem);
plan = repmat(decode_single_bomb(x(1:4),1,1,problem,1),1,3);
for u = 1:3
    idx = (u-1)*4+(1:4);
    plan(u) = decode_single_bomb(x(idx),u,1,problem,1);
end
end

function score = q4_objective(x,decoder,problem,cfg)
evaluation = evaluate_plan(decoder(x),problem,cfg,1,'fast');
score = evaluation.auxiliaryScore;
end

function [initialPoints,initialTable] = generate_q4_initial_points(problem,cfg,objective)
seedSets=cell(1,3);
for u=1:3
    seeds=generate_geometry_seeds(problem,u,1);
    scores=-inf(size(seeds,1),1);
    for k=1:size(seeds,1)
        e=evaluate_plan(decode_single_bomb(seeds(k,:),u,1,problem,1),problem,cfg,1,'fast');
        scores(k)=e.auxiliaryScore;
    end
    [~,order]=sort(scores,'descend');
    keep=order(1:min(cfg.q4.seedPerUav,numel(order)));
    seedSets{u}=seeds(keep,:);
end

counts=cellfun(@(x)size(x,1),seedSets);
allCount=prod(counts);
candidateCount=min(allCount,max(cfg.q4.initialCombinationCount*5,cfg.q4.initialCombinationCount));
rng(cfg.masterSeed+404,'twister');
indices=ones(1,3);
indexRows=indices;
while size(indexRows,1)<candidateCount
    row=[randi(counts(1)),randi(counts(2)),randi(counts(3))];
    indexRows(end+1,:)=row; %#ok<AGROW>
    indexRows=unique(indexRows,'rows','stable');
end
candidatePoints=zeros(size(indexRows,1),12);
for k=1:size(indexRows,1)
    candidatePoints(k,:)=[seedSets{1}(indexRows(k,1),:),seedSets{2}(indexRows(k,2),:), ...
        seedSets{3}(indexRows(k,3),:)];
end
scores=evaluate_population(candidatePoints,objective,false);
[scores,order]=sort(scores,'descend');
keep=order(1:min(cfg.q4.initialCombinationCount,numel(order)));
initialPoints=candidatePoints(keep,:);
initialTable=table((1:numel(keep))',scores(1:numel(keep)), ...
    'VariableNames',{'Rank','Score'});
end

function [x,history] = block_refine_q4(x,problem,cfg,objective)
rows=zeros(cfg.q4.blockCycles*3,4); row=0;
for cycle=1:cfg.q4.blockCycles
    for u=1:3
        idx=(u-1)*4+(1:4);
        lower=[-pi,70,0,0]; upper=[pi,140,problem.hitTime(1),problem.maxDelay(u)];
        repair=@(z)repair_single_bomb_decision(z,u,1,problem);
        blockObjective=@(z)objective(replace_block(x,idx,z));
        [~,options]=build_solver_options(cfg,400+cycle*10+u);
        options.populationSize=cfg.q4.blockPopulation;
        options.minPopulationSize=min(8,options.populationSize);
        options.maxIterations=cfg.q4.blockIterations;
        options.maxStallIterations=max(4,round(cfg.q4.blockIterations/3));
        options.debug=false;
        seeds=generate_geometry_seeds(problem,u,1);
        options.initialPoints=[x(idx);seeds(1:min(size(seeds,1),options.populationSize-1),:)];
        [bestBlock,bestScore]=adaptive_de(blockObjective,lower,upper,options,repair);
        x(idx)=bestBlock;
        row=row+1; rows(row,:)=[cycle,u,bestScore,objective(x)];
    end
end
rows=rows(1:row,:);
history=array2table(rows,'VariableNames',{'Cycle','UavIdx','BlockBestScore','JointScore'});
end

function y = replace_block(x,idx,z)
y=x; y(idx)=z;
end
