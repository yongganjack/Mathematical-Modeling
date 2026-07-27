function output = main()
%MAIN 问题5一键运行入口；在 MATLAB 编辑器中直接点击 Run。

clear; clc;
[projectRoot, questionDir] = prepare_entry_path();
defaultConfig = 'competition'; % 正式高预算计算时改为 'competition'
if strcmpi(defaultConfig, 'competition')
    cfg = competition_config();
else
    cfg = quick_config();
end
fprintf('[入口] 正在运行：%s\n', mfilename('fullpath'));
fprintf('[入口] 配置文件：%s\n', fullfile(projectRoot, 'configs', [cfg.name, '_config.m']));
output = run_question5(cfg);
save_question5_figures(output, projectRoot);
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

function save_question5_figures(output,projectRoot)
picDir=fullfile(projectRoot,'pic');if ~isfolder(picDir),mkdir(picDir);end;t=inline_theme_q5();p=load_problem_data();b=derive_bombs(output.plan,p);
fig=figure('Visible','off','Color','w','Position',[60,60,1100,700]);ax=axes(fig);inline_spatial_q5(ax,p,output.evaluation,b,t);title(ax,sprintf('Q5：五机 %d 弹对 M1--M3 的协同空间策略',numel(output.plan)));inline_export_q5(fig,fullfile(picDir,'Q5_multi_target_spatial.png'));close(fig);
fig=figure('Visible','off','Color','w','Position',[60,60,1250,650]);tl=tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');ax=nexttile(tl);inline_intervals_q5(ax,output.evaluation,t,'(a) 分目标有效遮蔽时段');ax=nexttile(tl);d=[output.evaluation.missiles.duration];h=bar(ax,1:3,d,'FaceColor','flat');h.CData=t.missile;for k=1:3,text(ax,k,d(k)+.3,sprintf('%.3f',d(k)),'HorizontalAlignment','center','FontName',t.font);end;yline(ax,output.evaluation.minimumDuration,'--k');text(ax,1.05,output.evaluation.minimumDuration+.55,sprintf('最短保障：%.3f s',output.evaluation.minimumDuration),'FontName',t.font);xticks(ax,1:3);xticklabels(ax,{'M1','M2','M3'});grid(ax,'on');xlabel(ax,'来袭导弹');ylabel(ax,'有效遮蔽时长 T (s)');title(ax,sprintf('(b) 分目标时长（总计 %.3f s）',output.evaluation.totalDuration));inline_style_q5(ax,t);inline_export_q5(fig,fullfile(picDir,'Q5_target_intervals_duration.png'));close(fig);
fig=figure('Visible','off','Color','w','Position',[60,60,1400,680]);tl=tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');inline_stage_q5(nexttile(tl),output.history.stage1,t.uav(1,:),'(a) 阶段1：总时长筛选','J_{sum} (s)',t);inline_stage_q5(nexttile(tl),output.history.stage2,t.uav(2,:),'(b) 阶段2：最短时长保障','J_{min} (s)',t);inline_stage_q5(nexttile(tl),output.history.refine,t.uav(3,:),'(c) 连续变量精修','J_{sum} (s)',t);title(tl,'Q5：三阶段优化收敛（各阶段目标独立量纲）','FontName',t.font,'FontWeight','bold');inline_export_q5(fig,fullfile(picDir,'Q5_three_stage_convergence.png'));close(fig);
end
function t=inline_theme_q5(),t.font='Microsoft YaHei';t.bg=[.985,.985,.985];t.uav=[0,.45,.70;.90,.62,0;0,.62,.45;.80,.47,.65;.34,.71,.91];t.missile=[0,.45,.70;.90,.62,0;.80,.47,.65];end
function inline_spatial_q5(ax,p,e,b,t),hold(ax,'on');grid(ax,'on');view(ax,3);[x,y,z]=cylinder(p.target.radius,30);surf(ax,x+p.target.center(1),y+p.target.center(2),z*p.target.height,'FaceColor',[.3,.3,.3],'FaceAlpha',.35,'EdgeColor','none','DisplayName','真目标');scatter3(ax,0,0,0,55,'kx','LineWidth',1.8,'DisplayName','假目标');for j=1:numel(e.missiles),m=e.missiles(j).missileIdx;plot3(ax,[p.missileInit(m,1),0],[p.missileInit(m,2),0],[p.missileInit(m,3),0],'--','Color',t.missile(m,:),'LineWidth',1.5,'DisplayName',char(p.missileIds(m)));end;for k=1:numel(b),q=b(k);c=t.uav(q.uavIdx,:);u=p.uavInit(q.uavIdx,:);plot3(ax,[u(1),q.releasePoint(1)],[u(2),q.releasePoint(2)],[u(3),q.releasePoint(3)],'-','Color',c,'LineWidth',1.7);plot3(ax,[q.releasePoint(1),q.burstPoint(1)],[q.releasePoint(2),q.burstPoint(2)],[q.releasePoint(3),q.burstPoint(3)],':','Color',c,'LineWidth',1.4);scatter3(ax,q.releasePoint(1),q.releasePoint(2),q.releasePoint(3),40,c,'o','filled');scatter3(ax,q.burstPoint(1),q.burstPoint(2),q.burstPoint(3),55,c,'^','filled');end;xlabel(ax,'X (m)');ylabel(ax,'Y (m)');zlabel(ax,'Z (m)');legend(ax,'Location','best');inline_style_q5(ax,t);end
function inline_intervals_q5(ax,e,t,titleText),hold(ax,'on');for j=1:numel(e.missiles),m=e.missiles(j);for k=1:size(m.intervals,1),rectangle(ax,'Position',[m.intervals(k,1),j-.28,diff(m.intervals(k,:)),.56],'FaceColor',t.missile(m.missileIdx,:),'EdgeColor','none');end;text(ax,max([m.intervals(:,2);0])+.2,j,sprintf('%.3f s',m.duration),'FontName',t.font);end;yticks(ax,1:numel(e.missiles));yticklabels(ax,arrayfun(@(m)sprintf('M%d',m.missileIdx),e.missiles,'UniformOutput',false));xlabel(ax,'时间 t (s)');ylabel(ax,'来袭导弹');title(ax,titleText);grid(ax,'on');inline_style_q5(ax,t);end
function inline_stage_q5(ax,h,c,titleText,yText,t),plot(ax,h.Iteration,h.BestScore,'Color',c,'LineWidth',1.6);grid(ax,'on');xlabel(ax,'迭代次数 k');ylabel(ax,yText);title(ax,titleText);inline_style_q5(ax,t);end
function inline_style_q5(ax,t),set(ax,'FontName',t.font,'FontSize',10,'LineWidth',.8,'Box','on','Color',t.bg);end
function inline_export_q5(fig,file),exportgraphics(fig,file,'Resolution',300,'BackgroundColor','white');fprintf('[Q5绘图] 已保存：%s\n',file);end
