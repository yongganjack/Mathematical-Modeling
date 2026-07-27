function monitor = create_live_convergence_monitor(questionName)
%CREATE_LIVE_CONVERGENCE_MONITOR 创建优化过程实时科研级收敛窗口。

palette = [0,114,178; 213,94,0; 0,158,115; 204,121,167; 230,159,0; 86,180,233]/255;
fig = figure('Name',[questionName,' - Live optimization'], 'NumberTitle','off', ...
    'Color','w','Position',[80,80,1180,720],'Visible','on');
layout = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
title(layout,[questionName,' optimization convergence'],'FontName','Arial','FontWeight','bold');

axScore = nexttile(layout,1); hold(axScore,'on'); grid(axScore,'on');
xlabel(axScore,'Iteration'); ylabel(axScore,'Search score'); title(axScore,'Best and mean score');
axDiversity = nexttile(layout,2); hold(axDiversity,'on'); grid(axDiversity,'on');
xlabel(axDiversity,'Iteration'); ylabel(axDiversity,'Normalized diversity'); title(axDiversity,'Population diversity');
axEvaluations = nexttile(layout,3); hold(axEvaluations,'on'); grid(axEvaluations,'on');
xlabel(axEvaluations,'Iteration'); ylabel(axEvaluations,'Function evaluations'); title(axEvaluations,'Evaluation budget');
axPopulation = nexttile(layout,4); hold(axPopulation,'on'); grid(axPopulation,'on');
xlabel(axPopulation,'Iteration'); ylabel(axPopulation,'Population size'); title(axPopulation,'Population schedule');
set([axScore,axDiversity,axEvaluations,axPopulation],'FontName','Arial','FontSize',10, ...
    'LineWidth',0.8,'Box','on','Color',[0.985,0.985,0.985]);

data.algorithms = struct();
data.palette = palette;
data.axes = [axScore,axDiversity,axEvaluations,axPopulation];
setappdata(fig,'LiveConvergenceData',data);
monitor.figure = fig;
monitor.callback = @(state) update_live_monitor(fig,state);
end

function update_live_monitor(fig,state)
if ~isgraphics(fig)
    return;
end
data = getappdata(fig,'LiveConvergenceData');
key = matlab.lang.makeValidName(char(state.algorithm));
if ~isfield(data.algorithms,key)
    colorIdx = mod(numel(fieldnames(data.algorithms)),size(data.palette,1))+1;
    color = data.palette(colorIdx,:);
    alg.label = char(state.algorithm);
    alg.iteration = [];
    alg.best = [];
    alg.mean = [];
    alg.diversity = [];
    alg.evaluations = [];
    alg.population = [];
    alg.bestLine = animatedline(data.axes(1),'Color',color,'LineWidth',2.0, ...
        'DisplayName',[alg.label,' best']);
    alg.meanLine = animatedline(data.axes(1),'Color',color,'LineWidth',1.1, ...
        'LineStyle','--','DisplayName',[alg.label,' mean']);
    alg.diversityLine = animatedline(data.axes(2),'Color',color,'LineWidth',1.7, ...
        'DisplayName',alg.label);
    alg.evaluationLine = animatedline(data.axes(3),'Color',color,'LineWidth',1.7, ...
        'DisplayName',alg.label);
    alg.populationLine = animatedline(data.axes(4),'Color',color,'LineWidth',1.7, ...
        'DisplayName',alg.label);
    data.algorithms.(key) = alg;
    legend(data.axes(1),'Location','best');
    legend(data.axes(2),'Location','best');
    legend(data.axes(3),'Location','best');
    legend(data.axes(4),'Location','best');
end
alg = data.algorithms.(key);
addpoints(alg.bestLine,state.iteration,state.bestScore);
addpoints(alg.meanLine,state.iteration,state.meanScore);
addpoints(alg.diversityLine,state.iteration,state.diversity);
addpoints(alg.evaluationLine,state.iteration,state.evaluations);
addpoints(alg.populationLine,state.iteration,state.populationSize);
data.algorithms.(key) = alg;
setappdata(fig,'LiveConvergenceData',data);
drawnow limitrate;
end
