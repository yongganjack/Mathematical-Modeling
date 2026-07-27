function output = main()
%MAIN 问题2一键运行入口；在 MATLAB 编辑器中直接点击 Run。

clear; clc;
close all;
[projectRoot, questionDir] = prepare_entry_path();
defaultConfig = 'competition'; % 正式高预算计算时改为 'competition'
if strcmpi(defaultConfig, 'competition')
    cfg = competition_config();
else
    cfg = quick_config();
end
fprintf('[入口] 正在运行：%s\n', mfilename('fullpath'));
fprintf('[入口] 配置文件：%s\n', fullfile(projectRoot, 'configs', [cfg.name, '_config.m']));
output = run_question2(cfg);
save_question2_figures(output, projectRoot, cfg);
end

function [projectRoot, questionDir] = prepare_entry_path()
questionDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(questionDir);
configDir = fullfile(projectRoot, 'configs');
commonDir = fullfile(projectRoot, 'common');
if ~isfile(fullfile(configDir, 'quick_config.m'))
    error('SmokeProject:ConfigNotFound', '未找到配置文件：%s', fullfile(configDir, 'quick_config.m'));
end
addpath(configDir, '-begin');
addpath(genpath(commonDir), '-begin');
addpath(questionDir, '-begin');
rehash path;
end

function save_question2_figures(output, projectRoot, cfg)
picDir=fullfile(projectRoot,'pic'); if ~isfolder(picDir),mkdir(picDir);end; theme=inline_theme_q2(); problem=load_problem_data(); bombs=derive_bombs(output.plan,problem);
q1.plan=struct('uavIdx',1,'bombId',1,'targetId',1,'theta',pi,'speed',120,'releaseTime',1.5,'delay',3.6); q1.evaluation=evaluate_plan(q1.plan,problem,cfg,1,'verify');
fig=figure('Visible','off','Color','w','Position',[60,60,1250,720]); tl=tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact'); ax=nexttile(tl); inline_spatial_q2(ax,problem,output.evaluation,bombs,theme,[0,.45,.70]); q1b=derive_bombs(q1.plan,problem); inline_flight_q2(ax,problem,q1b,[.45,.45,.45]); title(ax,'(a) 给定策略与优化策略的空间对比'); ax=nexttile(tl); axis(ax,'off'); text(ax,.03,.78,'(b) 决策变量对比','Units','normalized','FontName',theme.font,'FontSize',14,'FontWeight','bold'); text(ax,.03,.61,{sprintf('Q1 给定：(%.2f deg, %.2f m/s, %.3f s, %.3f s)',rad2deg(q1.plan.theta),q1.plan.speed,q1.plan.releaseTime,q1.plan.releaseTime+q1.plan.delay);sprintf('Q2 优化：(%.2f deg, %.2f m/s, %.3f s, %.3f s)',rad2deg(output.plan.theta),output.plan.speed,output.plan.releaseTime,output.plan.releaseTime+output.plan.delay)},'Units','normalized','FontName',theme.font,'FontSize',11,'VerticalAlignment','top'); text(ax,.03,.25,sprintf('优化后有效遮蔽时长：%.3f s\n相对 Q1 提升：%.3f s',output.evaluation.totalDuration,output.evaluation.totalDuration-q1.evaluation.totalDuration),'Units','normalized','FontName',theme.font,'FontSize',14); inline_export_q2(fig,fullfile(picDir,'Q2_strategy_comparison.png'));close(fig);
fig=figure('Visible','off','Color','w','Position',[60,60,1250,650]); tl=tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact'); ax=nexttile(tl); hold(ax,'on'); inline_history_q2(ax,fullfile(output.runDir,'optimization_history_pso.csv'),'PSO',theme.uav(1,:)); inline_history_q2(ax,fullfile(output.runDir,'optimization_history_de.csv'),'DE',theme.uav(2,:)); grid(ax,'on'); xlabel(ax,'迭代次数 k');ylabel(ax,'历史最优评分');title(ax,'(a) 优化收敛过程');legend(ax,'Location','best');inline_style_q2(ax,theme); ax=nexttile(tl); inline_intervals_q2(ax,output.evaluation,theme,'(b) 最终有效遮蔽时段');inline_export_q2(fig,fullfile(picDir,'Q2_convergence_intervals.png'));close(fig);
end
function theme=inline_theme_q2(), theme.font='Microsoft YaHei';theme.bg=[.985,.985,.985];theme.uav=[0,.45,.70;.90,.62,0;0,.62,.45;.80,.47,.65;.34,.71,.91];theme.missile=[0,.45,.70;.90,.62,0;.80,.47,.65];end
function inline_spatial_q2(ax,p,e,bombs,theme,flightColor), hold(ax,'on');grid(ax,'on');view(ax,3);[x,y,z]=cylinder(p.target.radius,30);surf(ax,x+p.target.center(1),y+p.target.center(2),z*p.target.height,'FaceColor',[.3,.3,.3],'FaceAlpha',.35,'EdgeColor','none');scatter3(ax,0,0,0,55,'kx','LineWidth',1.8);for j=1:numel(e.missiles),m=e.missiles(j).missileIdx;plot3(ax,[p.missileInit(m,1),0],[p.missileInit(m,2),0],[p.missileInit(m,3),0],'--','Color',theme.missile(m,:),'LineWidth',1.5);end;inline_flight_q2(ax,p,bombs,flightColor);xlabel(ax,'X (m)');ylabel(ax,'Y (m)');zlabel(ax,'Z (m)');inline_style_q2(ax,theme);end
function inline_flight_q2(ax,p,bombs,c),for k=1:numel(bombs),b=bombs(k);u=p.uavInit(b.uavIdx,:);plot3(ax,[u(1),b.releasePoint(1)],[u(2),b.releasePoint(2)],[u(3),b.releasePoint(3)],'-','Color',c,'LineWidth',1.8);plot3(ax,[b.releasePoint(1),b.burstPoint(1)],[b.releasePoint(2),b.burstPoint(2)],[b.releasePoint(3),b.burstPoint(3)],':','Color',c,'LineWidth',1.4);scatter3(ax,b.burstPoint(1),b.burstPoint(2),b.burstPoint(3),60,c,'^','filled');end;end
function inline_history_q2(ax,file,label,c),t=readtable(file);plot(ax,t.Iteration,t.BestScore,'Color',c,'LineWidth',1.6,'DisplayName',label);end
function inline_intervals_q2(ax,e,theme,titleText),hold(ax,'on');for j=1:numel(e.missiles),m=e.missiles(j);for k=1:size(m.intervals,1),rectangle(ax,'Position',[m.intervals(k,1),j-.28,diff(m.intervals(k,:)),.56],'FaceColor',theme.missile(m.missileIdx,:),'EdgeColor','none');end;text(ax,max([m.intervals(:,2);0])+.2,j,sprintf('%.3f s',m.duration),'FontName',theme.font);end;yticks(ax,1:numel(e.missiles));yticklabels(ax,arrayfun(@(m)sprintf('M%d',m.missileIdx),e.missiles,'UniformOutput',false));xlabel(ax,'时间 t (s)');ylabel(ax,'来袭导弹');title(ax,titleText);grid(ax,'on');inline_style_q2(ax,theme);end
function inline_style_q2(ax,theme),set(ax,'FontName',theme.font,'FontSize',10,'LineWidth',.8,'Box','on','Color',theme.bg);end
function inline_export_q2(fig,file),exportgraphics(fig,file,'Resolution',300,'BackgroundColor','white');fprintf('[Q2绘图] 已保存：%s\n',file);end
