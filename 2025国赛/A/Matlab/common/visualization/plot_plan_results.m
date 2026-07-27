function fig = plot_plan_results(problem, plan, evaluation, runDir, questionName, cfg)
%PLOT_PLAN_RESULTS 显示科研级多面板结果图；默认不导出文件。

if nargin < 6
    cfg.saveFigures = false;
end
bombs = derive_bombs(plan,problem);
palette = [0,114,178;213,94,0;0,158,115;204,121,167;230,159,0;86,180,233]/255;
fig = figure('Name',[questionName,' - Scientific summary'],'NumberTitle','off', ...
    'Visible','on','Color','w','Position',[40,40,1320,800]);
layout = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
title(layout,[questionName,' scientific visualization'],'FontName','Arial','FontWeight','bold');

ax = nexttile(layout,1); hold(ax,'on'); grid(ax,'on'); view(ax,3);
[xc,yc,zc] = cylinder(problem.target.radius,36); zc=zc.*problem.target.height;
surf(ax,xc+problem.target.center(1),yc+problem.target.center(2),zc+problem.target.center(3), ...
    'FaceColor',[0.25,0.25,0.25],'FaceAlpha',0.42,'EdgeColor','none','DisplayName','True target');
scatter3(ax,0,0,0,55,'k','x','LineWidth',1.8,'DisplayName','Decoy target');
for j=1:numel(evaluation.missiles)
    idx=evaluation.missiles(j).missileIdx;
    p=[problem.missileInit(idx,:);0,0,0];
    plot3(ax,p(:,1),p(:,2),p(:,3),'--','Color',palette(j,:),'LineWidth',1.5, ...
        'DisplayName',char(problem.missileIds(idx)));
end
for k=1:numel(bombs)
    u0=problem.uavInit(bombs(k).uavIdx,:);
    p=[u0;bombs(k).releasePoint;bombs(k).burstPoint];
    color=palette(mod(k-1,size(palette,1))+1,:);
    plot3(ax,p(:,1),p(:,2),p(:,3),'-','Color',color,'LineWidth',1.8,'HandleVisibility','off');
    scatter3(ax,bombs(k).releasePoint(1),bombs(k).releasePoint(2),bombs(k).releasePoint(3), ...
        42,color,'o','filled','HandleVisibility','off');
    scatter3(ax,bombs(k).burstPoint(1),bombs(k).burstPoint(2),bombs(k).burstPoint(3), ...
        62,color,'^','filled','DisplayName',sprintf('%s smoke %d',problem.uavIds(bombs(k).uavIdx),bombs(k).bombId));
end
xlabel(ax,'X (m)');ylabel(ax,'Y (m)');zlabel(ax,'Z (m)');title(ax,'(a) Three-dimensional trajectories');
legend(ax,'Location','best'); axis(ax,'vis3d');

ax=nexttile(layout,2); hold(ax,'on'); grid(ax,'on');
for j=1:numel(evaluation.missiles)
    intervals=evaluation.missiles(j).intervals;
    for r=1:size(intervals,1)
        rectangle(ax,'Position',[intervals(r,1),j-0.31,intervals(r,2)-intervals(r,1),0.62], ...
            'FaceColor',palette(j,:),'EdgeColor','none');
    end
end
yticks(ax,1:numel(evaluation.missiles));
yticklabels(ax,arrayfun(@(x)sprintf('M%d',x.missileIdx),evaluation.missiles,'UniformOutput',false));
xlabel(ax,'Time (s)');ylabel(ax,'Missile');title(ax,'(b) Effective shielding intervals');

ax=nexttile(layout,3); hold(ax,'on'); grid(ax,'on'); yline(ax,0,'k--','Shielding boundary');
for j=1:numel(evaluation.missiles)
    plot(ax,evaluation.missiles(j).timeGrid,evaluation.missiles(j).margin,'Color',palette(j,:), ...
        'LineWidth',1.6,'DisplayName',sprintf('M%d',evaluation.missiles(j).missileIdx));
end
xlabel(ax,'Time (s)');ylabel(ax,'Worst-case margin (m)');title(ax,'(c) Line-of-sight shielding margin');
legend(ax,'Location','best');

ax=nexttile(layout,4); hold(ax,'on'); grid(ax,'on');
for j=1:numel(evaluation.missiles)
    plot(ax,evaluation.missiles(j).timeGrid,100*evaluation.missiles(j).coverageRatio, ...
        'Color',palette(j,:),'LineWidth',1.6,'DisplayName',sprintf('M%d',evaluation.missiles(j).missileIdx));
end
yline(ax,100,'k--','Complete coverage'); ylim(ax,[0,105]);
xlabel(ax,'Time (s)');ylabel(ax,'Visible-point coverage (%)');title(ax,'(d) Target visibility coverage');
legend(ax,'Location','best');
set(findall(fig,'Type','axes'),'FontName','Arial','FontSize',10,'LineWidth',0.8,'Box','on','Color',[0.985,0.985,0.985]);
drawnow;
if isfield(cfg,'saveFigures') && cfg.saveFigures
    exportgraphics(fig,fullfile(runDir,'scientific_summary.png'),'Resolution',300);
end
end
