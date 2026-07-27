function output = main()
%MAIN 问题3一键运行入口；在 MATLAB 编辑器中直接点击 Run。

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
output = run_question3(cfg);
save_question3_figures(output, projectRoot);
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

function save_question3_figures(output,projectRoot)
picDir=fullfile(projectRoot,'pic');if ~isfolder(picDir),mkdir(picDir);end;t=inline_theme_q3();p=load_problem_data();b=derive_bombs(output.plan,p);
fig=figure('Visible','off','Color','w','Position',[60,60,1100,700]);ax=axes(fig);inline_spatial_q3(ax,p,output.evaluation,b,t);title(ax,'Q3：FY1 三枚烟幕弹的协同空间策略');inline_export_q3(fig,fullfile(picDir,'Q3_three_bomb_spatial.png'));close(fig);
fig=figure('Visible','off','Color','w','Position',[60,60,1200,760]);tl=tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');ax=nexttile(tl);inline_intervals_q3(ax,output.evaluation,t,'(a) 遮蔽并集时段');ax=nexttile(tl);hold(ax,'on');grid(ax,'on');for k=1:numel(b),plot(ax,[b(k).burstTime,b(k).cloudEndTime],[k,k],'-','Color',t.uav(1,:),'LineWidth',6);xline(ax,b(k).burstTime,':','Color',[.3,.3,.3]);end;ylim(ax,[.5,numel(b)+.5]);yticks(ax,1:numel(b));xlabel(ax,'时间 t (s)');ylabel(ax,'烟幕弹编号');title(ax,'(b) 单弹有效云团时间窗');inline_style_q3(ax,t);inline_export_q3(fig,fullfile(picDir,'Q3_interval_contributions.png'));close(fig);
fig=figure('Visible','off','Color','w','Position',[60,60,1200,650]);tl=tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');ax=nexttile(tl);hold(ax,'on');inline_hist_q3(ax,output.history.pso,'PSO',t.uav(1,:));inline_hist_q3(ax,output.history.de,'DE',t.uav(2,:));grid(ax,'on');legend(ax,'Location','best');xlabel(ax,'迭代次数 k');ylabel(ax,'历史最优评分');title(ax,'(a) PSO 与 DE 收敛');inline_style_q3(ax,t);ax=nexttile(tl);v=output.history.verification;v=sortrows(v,'TotalDuration','descend');v=v(1:min(8,height(v)),:);bar(ax,1:height(v),v.TotalDuration,'FaceColor',t.uav(3,:));grid(ax,'on');xlabel(ax,'按时长排序的前 8 个候选');ylabel(ax,'高精度遮蔽时长 T (s)');title(ax,'(b) 候选解高精度复核');inline_style_q3(ax,t);inline_export_q3(fig,fullfile(picDir,'Q3_optimizer_convergence.png'));close(fig);
end
function t=inline_theme_q3(),t.font='Microsoft YaHei';t.bg=[.985,.985,.985];t.uav=[0,.45,.70;.90,.62,0;0,.62,.45;.80,.47,.65;.34,.71,.91];t.missile=[0,.45,.70;.90,.62,0;.80,.47,.65];end
function inline_spatial_q3(ax,p,e,b,t),hold(ax,'on');grid(ax,'on');view(ax,3);[x,y,z]=cylinder(p.target.radius,30);surf(ax,x+p.target.center(1),y+p.target.center(2),z*p.target.height,'FaceColor',[.3,.3,.3],'FaceAlpha',.35,'EdgeColor','none','DisplayName','真目标');scatter3(ax,0,0,0,55,'kx','LineWidth',1.8,'DisplayName','假目标');for j=1:numel(e.missiles),m=e.missiles(j).missileIdx;plot3(ax,[p.missileInit(m,1),0],[p.missileInit(m,2),0],[p.missileInit(m,3),0],'--','Color',t.missile(m,:),'LineWidth',1.5,'DisplayName',char(p.missileIds(m)));end;for k=1:numel(b),q=b(k);c=t.uav(q.uavIdx,:);u=p.uavInit(q.uavIdx,:);plot3(ax,[u(1),q.releasePoint(1)],[u(2),q.releasePoint(2)],[u(3),q.releasePoint(3)],'-','Color',c,'LineWidth',1.7);plot3(ax,[q.releasePoint(1),q.burstPoint(1)],[q.releasePoint(2),q.burstPoint(2)],[q.releasePoint(3),q.burstPoint(3)],':','Color',c,'LineWidth',1.4);scatter3(ax,q.releasePoint(1),q.releasePoint(2),q.releasePoint(3),45,c,'o','filled');scatter3(ax,q.burstPoint(1),q.burstPoint(2),q.burstPoint(3),58,c,'^','filled');end;xlabel(ax,'X (m)');ylabel(ax,'Y (m)');zlabel(ax,'Z (m)');legend(ax,'Location','best');inline_style_q3(ax,t);end
function inline_intervals_q3(ax,e,t,titleText),hold(ax,'on');for j=1:numel(e.missiles),m=e.missiles(j);for k=1:size(m.intervals,1),rectangle(ax,'Position',[m.intervals(k,1),j-.28,diff(m.intervals(k,:)),.56],'FaceColor',t.missile(m.missileIdx,:),'EdgeColor','none');end;text(ax,max([m.intervals(:,2);0])+.2,j,sprintf('%.3f s',m.duration),'FontName',t.font);end;yticks(ax,1:numel(e.missiles));yticklabels(ax,arrayfun(@(m)sprintf('M%d',m.missileIdx),e.missiles,'UniformOutput',false));xlabel(ax,'时间 t (s)');ylabel(ax,'来袭导弹');title(ax,titleText);grid(ax,'on');inline_style_q3(ax,t);end
function inline_hist_q3(ax,h,label,c),plot(ax,h.Iteration,h.BestScore,'Color',c,'LineWidth',1.6,'DisplayName',label);end
function inline_style_q3(ax,t),set(ax,'FontName',t.font,'FontSize',10,'LineWidth',.8,'Box','on','Color',t.bg);end
function inline_export_q3(fig,file),exportgraphics(fig,file,'Resolution',300,'BackgroundColor','white');fprintf('[Q3绘图] 已保存：%s\n',file);end
