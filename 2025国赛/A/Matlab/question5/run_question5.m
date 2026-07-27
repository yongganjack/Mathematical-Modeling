function output = run_question5(cfg)
%RUN_QUESTION5 构造逐弹目标路线库，执行严格字典序路线选择和连续精修。

fprintf('\n========== 开始求解问题5（%s 配置）==========\n',cfg.name);
problem=load_problem_data();
runDir=create_run_directory(cfg,'question5');
try
    routeLibrary=generate_route_library(problem,cfg);
    routeCounts=cellfun(@numel,routeLibrary);
    fprintf('[问题5] 各无人机路线数量：%s。\n',mat2str(routeCounts));
    write_json_file(fullfile(runDir,'route_library.json'),routeLibrary);

    lowerBound=ones(1,5); upperBound=routeCounts;
    selectionRepair=@(x)repair_selection(x,routeCounts);
    stage1Objective=@(x)selection_score(x,routeLibrary,problem,cfg,1,-inf);
    [selectionOptions,~]=build_solver_options(cfg,500);
    selectionOptions.swarmSize=cfg.q5.selectionSwarmSize;
    selectionOptions.maxIterations=cfg.q5.selectionIterations;
    selectionOptions.maxStallIterations=max(5,round(cfg.q5.selectionIterations/4));
    selectionOptions.debug=cfg.debug;
    selectionOptions.algorithmLabel='Stage 1 PSO';
    selectionSeeds=build_selection_seeds(routeLibrary,cfg,stage1Objective);
    selectionOptions.initialPoints=selectionSeeds;
    if cfg.enablePlots && cfg.livePlots
        liveMonitor=create_live_convergence_monitor('Question 5');
        selectionOptions.liveCallback=liveMonitor.callback;
    end
    [stage1X,~,stage1History]=improved_pso(stage1Objective,lowerBound,upperBound, ...
        selectionOptions,selectionRepair);
    stage1Plan=decode_selection(stage1X,routeLibrary);
    stage1Fast=evaluate_plan(stage1Plan,problem,cfg,1:3,'fast');
    bestFastSum=stage1Fast.totalDuration;
    fprintf('[问题5] 第一阶段真实快速 J_sum=%.6f s。\n',bestFastSum);

    stage2Objective=@(x)selection_score(x,routeLibrary,problem,cfg,2,bestFastSum);
    selectionOptions.seed=selectionOptions.seed+1;
    selectionOptions.algorithmLabel='Stage 2 PSO';
    selectionOptions.initialPoints=[stage1X;selectionSeeds];
    [stage2X,~,stage2History]=improved_pso(stage2Objective,lowerBound,upperBound, ...
        selectionOptions,selectionRepair);
    stage2Plan=decode_selection(stage2X,routeLibrary);
    stage2Fast=evaluate_plan(stage2Plan,problem,cfg,1:3,'fast');
    if stage2Fast.totalDuration<bestFastSum-cfg.lexicographicTolerance
        warning('SmokeOptimization:LexicographicFallback','第二阶段未保持真实总时长容差，已回退第一阶段路线。');
        selectedPlan=stage1Plan;
    else
        selectedPlan=stage2Plan;
    end

    refineHistory=table(); refinedPlan=selectedPlan;
    if cfg.q5.continuousRefineIterations>0 && ~isempty(selectedPlan)
        [x0,refineLower,refineUpper,metadata]=pack_refine_decision(selectedPlan,problem);
        refineRepair=@(x)repair_refine_decision(x,metadata,problem);
        refineDecoder=@(x)decode_refine_decision(x,metadata,problem);
        refineObjective=@(x)refine_score(x,refineDecoder,problem,cfg);
        [~,deOptions]=build_solver_options(cfg,800);
        deOptions.populationSize=max(8,min(cfg.q5.routePopulation,4*numel(x0)));
        deOptions.minPopulationSize=min(8,deOptions.populationSize);
        deOptions.maxIterations=cfg.q5.continuousRefineIterations;
        deOptions.maxStallIterations=max(3,round(cfg.q5.continuousRefineIterations/2));
        deOptions.initialPoints=x0;
        if cfg.enablePlots && cfg.livePlots
            deOptions.liveCallback=liveMonitor.callback;
            deOptions.algorithmLabel='Refinement DE';
        end
        [refinedX,~,refineHistory]=adaptive_de(refineObjective,refineLower,refineUpper, ...
            deOptions,refineRepair);
        refinedPlan=refineDecoder(refinedX);
    end

    selectedVerify=evaluate_plan(selectedPlan,problem,cfg,1:3,'verify');
    refinedVerify=evaluate_plan(refinedPlan,problem,cfg,1:3,'verify');
    if lexicographically_better(refinedVerify,selectedVerify,cfg.lexicographicTolerance)
        plan=refinedPlan; evaluation=refinedVerify; refineAccepted=true;
    else
        plan=selectedPlan; evaluation=selectedVerify; refineAccepted=false;
    end

    history.stage1=stage1History; history.stage2=stage2History; history.refine=refineHistory;
    save_run_bundle(runDir,'question5',cfg,problem,plan,evaluation,history);
    selectionSummary.stage1Selection=round(stage1X);
    selectionSummary.stage2Selection=round(stage2X);
    selectionSummary.refineAccepted=refineAccepted;
    selectionSummary.routeCounts=routeCounts;
    selectionSummary.T=[evaluation.missiles.duration];
    selectionSummary.Jsum=evaluation.totalDuration;
    selectionSummary.Jmin=evaluation.minimumDuration;
    write_json_file(fullfile(runDir,'route_selection.json'),selectionSummary);

    excelDir=fullfile(runDir,'excel');mkdir(excelDir);
    template=fullfile(cfg.projectRoot,'00_赛题资料','附件','result3.xlsx');
    validation=export_result_workbook(5,template,fullfile(excelDir,'result3.xlsx'),plan,problem,cfg);
    write_json_file(fullfile(excelDir,'export_validation.json'),validation);
    if cfg.enablePlots && cfg.showStaticFigures
        plot_plan_results(problem,plan,evaluation,runDir,'Question 5',cfg);
    end
    if cfg.enablePlots && cfg.livePlots
        animate_plan_3d(problem,plan,evaluation,cfg,'Question 5');
    end
    print_plan_console_summary(problem,plan,evaluation,'问题5');
    fprintf('[问题5] 结果目录：%s\n',runDir);
catch ME
    fprintf(2,'[问题5异常] 求解失败：%s\n',ME.message);
    rethrow(ME);
end
output=struct('runDir',runDir,'plan',plan,'evaluation',evaluation, ...
    'routeLibrary',{routeLibrary},'history',history);
end

function routeLibrary=generate_route_library(problem,cfg)
routeLibrary=cell(1,5); patterns=route_patterns(); emptyPlan=empty_plan();
for u=1:5
    geometryBanks=build_geometry_banks(problem,cfg,u);
    routes=struct('id',1,'uavIdx',u,'targetPattern',[],'bombCount',0, ...
        'plan',emptyPlan,'fastDuration',0,'minimumDuration',0, ...
        'missileDurations',zeros(1,3),'score',0);
    nextId=2;
    for p=1:numel(patterns)
        targetIds=patterns{p}; bombCount=numel(targetIds);
        lower=[-pi,70,zeros(1,bombCount),zeros(1,bombCount)];
        upper=[pi,140,problem.hitTime(targetIds).',repmat(problem.maxDelay(u),1,bombCount)];
        repair=@(x)repair_mixed_route_decision(x,u,targetIds,problem);
        decoder=@(x)decode_mixed_route(x,u,targetIds,problem);
        objective=@(x)route_score(x,decoder,problem,cfg);
        [~,deOptions]=build_solver_options(cfg,100*u+p);
        deOptions.populationSize=cfg.q5.routePopulation;
        deOptions.minPopulationSize=min(6,deOptions.populationSize);
        if numel(unique(targetIds))>1
            deOptions.maxIterations=cfg.q5.mixedRouteIterations;
        else
            deOptions.maxIterations=cfg.q5.routeIterations;
        end
        deOptions.maxStallIterations=max(3,round(deOptions.maxIterations/2));
        deOptions.debug=false;
        initialPoints=generate_route_pattern_seeds(problem,cfg,u,targetIds,objective,geometryBanks);
        deOptions.initialPoints=initialPoints;
        [bestX,~,~,candidates]=adaptive_de(objective,lower,upper,deOptions,repair);
        extraCount=max(0,cfg.q5.routesPerUavMissile-1);
        candidateRows=[bestX;candidates.x(1:min(extraCount,size(candidates.x,1)),:)];
        candidateRows=unique(round(candidateRows,10),'rows','stable');
        for c=1:size(candidateRows,1)
            plan=decoder(candidateRows(c,:));
            fastEvaluation=evaluate_plan(plan,problem,cfg,1:3,'fast');
            routes(end+1)=struct('id',nextId,'uavIdx',u,'targetPattern',targetIds, ...
                'bombCount',bombCount,'plan',plan,'fastDuration',fastEvaluation.totalDuration, ...
                'minimumDuration',fastEvaluation.minimumDuration, ...
                'missileDurations',[fastEvaluation.missiles.duration], ...
                'score',fastEvaluation.auxiliaryScore); %#ok<AGROW>
            nextId=nextId+1;
        end
    end
    if numel(routes)>cfg.q5.maxRoutesPerUav
        [~,order]=sort([routes(2:end).score],'descend');
        keep=order(1:min(cfg.q5.maxRoutesPerUav-1,numel(order)))+1;
        routes=routes([1,keep]);
    end
    for r=1:numel(routes),routes(r).id=r;end
    routeLibrary{u}=routes;
    positive=sum([routes.fastDuration]>0);
    fprintf('[问题5] %s 路线库：%d条，其中正时长路线%d条。\n',problem.uavIds(u),numel(routes),positive);
end
end

function banks=build_geometry_banks(problem,cfg,uavIdx)
banks=cell(1,3);
for targetId=1:3
    seeds=generate_geometry_seeds(problem,uavIdx,targetId);
    if isempty(seeds),banks{targetId}=seeds;continue;end
    scores=-inf(size(seeds,1),1);
    for s=1:size(seeds,1)
        e=evaluate_plan(decode_single_bomb(seeds(s,:),uavIdx,targetId,problem,1), ...
            problem,cfg,targetId,'fast');
        scores(s)=e.auxiliaryScore;
    end
    [~,order]=sort(scores,'descend');
    banks{targetId}=seeds(order(1:min(cfg.q5.seedPerPattern,numel(order))),:);
end
end

function patterns=route_patterns()
patterns={ [1],[2],[3], ...
    [1,1],[2,2],[3,3],[1,2],[2,1],[1,3],[3,1],[2,3],[3,2], ...
    [1,1,1],[2,2,2],[3,3,3], ...
    [1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1] };
end

function initialPoints=generate_route_pattern_seeds(problem,cfg,uavIdx,targetIds,objective,seedBanksByTarget)
bombCount=numel(targetIds); seedBank=cell(1,bombCount); anchors=zeros(0,4);
for k=1:bombCount
    seeds=seedBanksByTarget{targetIds(k)};
    if ~isempty(seeds),anchors=[anchors;seeds];end %#ok<AGROW>
    seedBank{k}=seeds;
end
if isempty(anchors)
    lower=[-pi,70,zeros(1,bombCount),zeros(1,bombCount)];
    upper=[pi,140,problem.hitTime(targetIds).',repmat(problem.maxDelay(uavIdx),1,bombCount)];
    initialPoints=stratified_population(cfg.q5.seedPerPattern,lower,upper);
    return;
end
rows=zeros(0,2+2*bombCount); spacings=[1,4,8];
for a=1:size(anchors,1)
    anchor=anchors(a,:);
    releases=zeros(1,bombCount);delays=zeros(1,bombCount);
    for k=1:bombCount
        bank=seedBank{k};
        if isempty(bank)
            releases(k)=max(0,(k-1)*4);delays(k)=min(2,problem.maxDelay(uavIdx));
        else
            metric=abs(wrap_angle(bank(:,1)-anchor(1)))/pi+abs(bank(:,2)-anchor(2))/70;
            [~,idx]=min(metric);releases(k)=bank(idx,3);delays(k)=bank(idx,4);
        end
    end
    for spacing=spacings
        raw=[anchor(1),anchor(2),releases+(0:bombCount-1)*spacing,delays];
        rows(end+1,:)=repair_mixed_route_decision(raw,uavIdx,targetIds,problem); %#ok<AGROW>
    end
end
rows=unique(round(rows,10),'rows','stable');
scores=evaluate_population(rows,objective,false);
[~,order]=sort(scores,'descend');
initialPoints=rows(order(1:min(cfg.q5.seedPerPattern,numel(order))),:);
end

function angle=wrap_angle(angle)
angle=mod(angle+pi,2*pi)-pi;
end

function score=route_score(x,decoder,problem,cfg)
evaluation=evaluate_plan(decoder(x),problem,cfg,1:3,'fast');
score=evaluation.auxiliaryScore;
end

function seeds=build_selection_seeds(routeLibrary,cfg,objective)
counts=cellfun(@numel,routeLibrary);rng(cfg.masterSeed+505,'twister');
sampleCount=max(cfg.q5.selectionSeedCount,cfg.q5.selectionSwarmSize);
rows=ones(sampleCount,5);
for u=1:5,rows(:,u)=randi(counts(u),sampleCount,1);end
% 加入各无人机独立最高真实总时长路线的组合。
bestRow=ones(1,5);
for u=1:5
    [~,bestRow(u)]=max([routeLibrary{u}.fastDuration]);
end
rows(1,:)=bestRow;
% 单导弹强化组合与三种轮换均衡组合，避免第二阶段停在J_min=0平台。
insertRow=2;
for targetId=1:3
    row=ones(1,5);
    for u=1:5
        durationMatrix=reshape([routeLibrary{u}.missileDurations],3,[]).';
        [~,row(u)]=max(durationMatrix(:,targetId));
    end
    rows(insertRow,:)=row;insertRow=insertRow+1;
end
for shift=0:2
    row=ones(1,5);
    for u=1:5
        targetId=mod(u-1+shift,3)+1;
        durationMatrix=reshape([routeLibrary{u}.missileDurations],3,[]).';
        [~,row(u)]=max(durationMatrix(:,targetId));
    end
    rows(insertRow,:)=row;insertRow=insertRow+1;
end
specialRows=unique(rows(1:insertRow-1,:),'rows','stable');
rows=unique(rows,'rows','stable');
scores=evaluate_population(rows,objective,false);
[~,order]=sort(scores,'descend');
topRows=rows(order(1:min(cfg.q5.selectionSwarmSize,numel(order))),:);
seeds=unique([specialRows;topRows],'rows','stable');
end

function x=repair_selection(x,routeCounts)
x=round(x);x=min(routeCounts,max(ones(size(x)),x));
end

function plan=decode_selection(x,routeLibrary)
x=repair_selection(x,cellfun(@numel,routeLibrary));plan=empty_plan();
for u=1:5
    piece=routeLibrary{u}(x(u)).plan;
    if ~isempty(piece),plan=[plan,piece];end %#ok<AGROW>
end
end

function score=selection_score(x,routeLibrary,problem,cfg,stage,bestSum)
evaluation=evaluate_plan(decode_selection(x,routeLibrary),problem,cfg,1:3,'fast');
if stage==1
    score=evaluation.totalDuration;
else
    deficit=bestSum-cfg.lexicographicTolerance-evaluation.totalDuration;
    if deficit>0,score=-1e3-100*deficit;
    else,score=evaluation.minimumDuration+1e-8*evaluation.totalDuration;end
end
end

function [x0,lower,upper,metadata]=pack_refine_decision(plan,problem)
x0=[];lower=[];upper=[];
metadata=struct('uavIdx',{},'targetIds',{},'bombCount',{},'startIndex',{},'length',{});
for u=1:5
    idx=find([plan.uavIdx]==u);if isempty(idx),continue;end
    [~,order]=sort([plan(idx).bombId]);idx=idx(order);bombCount=numel(idx);
    targetIds=[plan(idx).targetId];
    values=[plan(idx(1)).theta,plan(idx(1)).speed,[plan(idx).releaseTime],[plan(idx).delay]];
    startIndex=numel(x0)+1;x0=[x0,values]; %#ok<AGROW>
    lower=[lower,-pi,70,zeros(1,bombCount),zeros(1,bombCount)]; %#ok<AGROW>
    upper=[upper,pi,140,problem.hitTime(targetIds).',repmat(problem.maxDelay(u),1,bombCount)]; %#ok<AGROW>
    metadata(end+1)=struct('uavIdx',u,'targetIds',targetIds,'bombCount',bombCount, ...
        'startIndex',startIndex,'length',numel(values)); %#ok<AGROW>
end
end

function x=repair_refine_decision(x,metadata,problem)
for g=1:numel(metadata)
    idx=metadata(g).startIndex:(metadata(g).startIndex+metadata(g).length-1);
    x(idx)=repair_mixed_route_decision(x(idx),metadata(g).uavIdx,metadata(g).targetIds,problem);
end
end

function plan=decode_refine_decision(x,metadata,problem)
x=repair_refine_decision(x,metadata,problem);plan=empty_plan();
for g=1:numel(metadata)
    idx=metadata(g).startIndex:(metadata(g).startIndex+metadata(g).length-1);
    groupPlan=decode_mixed_route(x(idx),metadata(g).uavIdx,metadata(g).targetIds,problem);
    plan=[plan,groupPlan]; %#ok<AGROW>
end
end

function score=refine_score(x,decoder,problem,cfg)
evaluation=evaluate_plan(decoder(x),problem,cfg,1:3,'fast');
score=evaluation.totalDuration+1e-7*evaluation.minimumDuration;
end

function tf=lexicographically_better(a,b,tolerance)
if a.totalDuration>b.totalDuration+tolerance
    tf=true;
elseif abs(a.totalDuration-b.totalDuration)<=tolerance
    tf=a.minimumDuration>b.minimumDuration;
else
    tf=false;
end
end

function plan=empty_plan()
plan=struct('uavIdx',{},'bombId',{},'targetId',{},'theta',{},'speed',{},'releaseTime',{},'delay',{});
end
