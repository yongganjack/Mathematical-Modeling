function output = main()
%MAIN 问题1一键运行入口；在 MATLAB 编辑器中直接点击 Run。

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
output = run_question1(cfg);
save_question1_figures(output, projectRoot);
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

function save_question1_figures(output, projectRoot)
picDir = fullfile(projectRoot, 'pic'); if ~isfolder(picDir), mkdir(picDir); end
theme = inline_theme_q1(); problem = load_problem_data(); bombs = derive_bombs(output.plan, problem);
fig = figure('Visible','off','Color','w','Position',[60,60,1100,700]); ax = axes(fig);
inline_spatial_q1(ax, problem, output.evaluation, bombs, theme); title(ax,'Q1：无人机、烟幕弹与 M1 的空间关系');
inline_export_q1(fig, fullfile(picDir,'Q1_spatial_strategy.png')); close(fig);
missile = output.evaluation.missiles(1); fig = figure('Visible','off','Color','w','Position',[60,60,1200,760]);
tl = tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');
ax = nexttile(tl); hold(ax,'on'); plot(ax,missile.timeGrid,missile.margin,'Color',theme.missile(1,:),'LineWidth',1.5); yline(ax,0,'--k','遮蔽判定边界'); inline_shade_q1(ax,missile.intervals); grid(ax,'on'); xlabel(ax,'时间 t (s)'); ylabel(ax,'最小裕度 d (m)'); title(ax,'(a) 视线遮蔽裕度'); inline_style_q1(ax,theme);
ax = nexttile(tl); hold(ax,'on'); plot(ax,missile.timeGrid,100*missile.coverageRatio,'Color',theme.missile(1,:),'LineWidth',1.5); yline(ax,100,'--k','完全遮蔽阈值'); inline_shade_q1(ax,missile.intervals); ylim(ax,[0,105]); grid(ax,'on'); xlabel(ax,'时间 t (s)'); ylabel(ax,'可见点遮蔽率 C (%)'); title(ax,'(b) 目标可见点遮蔽率'); inline_style_q1(ax,theme);
title(tl,sprintf('Q1：有效遮蔽时长 %.3f s',missile.duration),'FontName',theme.font,'FontWeight','bold'); inline_export_q1(fig,fullfile(picDir,'Q1_shielding_diagnosis.png')); close(fig);
end

function theme = inline_theme_q1()
theme.font='Microsoft YaHei'; theme.bg=[.985,.985,.985]; theme.uav=[0,.45,.70;.90,.62,0;0,.62,.45;.80,.47,.65;.34,.71,.91]; theme.missile=[0,.45,.70;.90,.62,0;.80,.47,.65];
end
function inline_spatial_q1(ax,problem,evaluation,bombs,theme)
hold(ax,'on'); grid(ax,'on'); view(ax,3); [x,y,z]=cylinder(problem.target.radius,30); surf(ax,x+problem.target.center(1),y+problem.target.center(2),z*problem.target.height,'FaceColor',[.3,.3,.3],'FaceAlpha',.35,'EdgeColor','none','DisplayName','真目标'); scatter3(ax,0,0,0,55,'kx','LineWidth',1.8,'DisplayName','假目标');
for j=1:numel(evaluation.missiles), m=evaluation.missiles(j).missileIdx; plot3(ax,[problem.missileInit(m,1),0],[problem.missileInit(m,2),0],[problem.missileInit(m,3),0],'--','Color',theme.missile(m,:),'LineWidth',1.5,'DisplayName',char(problem.missileIds(m))); end
for k=1:numel(bombs), b=bombs(k); c=theme.uav(b.uavIdx,:); u=problem.uavInit(b.uavIdx,:); plot3(ax,[u(1),b.releasePoint(1)],[u(2),b.releasePoint(2)],[u(3),b.releasePoint(3)],'-','Color',c,'LineWidth',1.7,'HandleVisibility','off'); plot3(ax,[b.releasePoint(1),b.burstPoint(1)],[b.releasePoint(2),b.burstPoint(2)],[b.releasePoint(3),b.burstPoint(3)],':','Color',c,'LineWidth',1.4,'HandleVisibility','off'); scatter3(ax,b.releasePoint(1),b.releasePoint(2),b.releasePoint(3),45,c,'o','filled','HandleVisibility','off'); scatter3(ax,b.burstPoint(1),b.burstPoint(2),b.burstPoint(3),58,c,'^','filled','HandleVisibility','off'); end
xlabel(ax,'X (m)'); ylabel(ax,'Y (m)'); zlabel(ax,'Z (m)'); legend(ax,'Location','best'); inline_style_q1(ax,theme);
end
function inline_shade_q1(ax,intervals)
if isempty(intervals), return; end; yl=ylim(ax); for k=1:size(intervals,1), patch(ax,[intervals(k,1),intervals(k,2),intervals(k,2),intervals(k,1)],[yl(1),yl(1),yl(2),yl(2)],[.7,.85,.95],'FaceAlpha',.22,'EdgeColor','none','HandleVisibility','off'); end
end
function inline_style_q1(ax,theme), set(ax,'FontName',theme.font,'FontSize',10,'LineWidth',.8,'Box','on','Color',theme.bg); end
function inline_export_q1(fig,file), exportgraphics(fig,file,'Resolution',300,'BackgroundColor','white'); fprintf('[Q1绘图] 已保存：%s\n',file); end
