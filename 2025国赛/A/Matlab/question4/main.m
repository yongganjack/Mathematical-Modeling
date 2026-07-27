function output = main()
%MAIN 问题4一键运行入口；在 MATLAB 编辑器中直接点击 Run。

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
output = run_question4(cfg);
save_question4_figures(output, projectRoot);
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

function save_question4_figures(output,projectRoot)
picDir=fullfile(projectRoot,'pic');if ~isfolder(picDir),mkdir(picDir);end;t=inline_theme_q4();p=load_problem_data();b=derive_bombs(output.plan,p);
fig=figure('Visible','off','Color','w','Position',[60,60,1100,700]);ax=axes(fig);inline_spatial_q4(ax,p,output.evaluation,b,t);title(ax,'Q4：FY1--FY3 的协同空间部署');inline_export_q4(fig,fullfile(picDir,'Q4_multi_uav_spatial.png'));close(fig);
fig=figure('Visible','off','Color','w','Position',[60,60,1200,760]);tl=tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');ax=nexttile(tl);inline_intervals_q4(ax,output.evaluation,t,'(a) 总有效遮蔽时段');ax=nexttile(tl);hold(ax,'on');grid(ax,'on');for k=1:numel(b),c=t.uav(b(k).uavIdx,:);plot(ax,[b(k).releaseTime,b(k).burstTime],[k,k],'-','Color',c,'LineWidth',5,'DisplayName',char(p.uavIds(b(k).uavIdx)));scatter(ax,b(k).releaseTime,k,32,c,'o','filled','HandleVisibility','off');scatter(ax,b(k).burstTime,k,40,c,'^','filled','HandleVisibility','off');end;yticks(ax,1:numel(b));xlabel(ax,'时间 t (s)');ylabel(ax,'投放序号');title(ax,'(b) 投放—起爆时序');legend(ax,'Location','best');inline_style_q4(ax,t);inline_export_q4(fig,fullfile(picDir,'Q4_timing_contribution.png'));close(fig);
fig=figure('Visible','off','Color','w','Position',[60,60,1200,650]);tl=tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');ax=nexttile(tl);hold(ax,'on');inline_hist_q4(ax,output.history.pso,'PSO',t.uav(1,:));inline_hist_q4(ax,output.history.de,'DE',t.uav(2,:));grid(ax,'on');legend(ax,'Location','best');xlabel(ax,'迭代次数 k');ylabel(ax,'历史最优评分');title(ax,'(a) PSO 与 DE 收敛');inline_style_q4(ax,t);ax=nexttile(tl);x=[max(output.history.initial.Score),output.history.block.JointScore(end)];h=bar(ax,1:2,x,'FaceColor','flat');h.CData=[t.uav(3,:);t.uav(4,:)];for k=1:2,text(ax,k,x(k)+.12,sprintf('%.3f',x(k)),'HorizontalAlignment','center','FontName',t.font);end;xticks(ax,1:2);xticklabels(ax,{'最优初始几何候选','分块精修后'});ylabel(ax,'联合评分');title(ax,'(b) 初值与分块精修结果');grid(ax,'on');inline_style_q4(ax,t);inline_export_q4(fig,fullfile(picDir,'Q4_multistage_convergence.png'));close(fig);
end
function t=inline_theme_q4(),t.font='Microsoft YaHei';t.bg=[.985,.985,.985];t.uav=[0,.45,.70;.90,.62,0;0,.62,.45;.80,.47,.65;.34,.71,.91];t.missile=[0,.45,.70;.90,.62,0;.80,.47,.65];end
function inline_spatial_q4(ax,p,e,b,t),hold(ax,'on');grid(ax,'on');view(ax,3);[x,y,z]=cylinder(p.target.radius,30);surf(ax,x+p.target.center(1),y+p.target.center(2),z*p.target.height,'FaceColor',[.3,.3,.3],'FaceAlpha',.35,'EdgeColor','none','DisplayName','真目标');scatter3(ax,0,0,0,55,'kx','LineWidth',1.8,'DisplayName','假目标');for j=1:numel(e.missiles),m=e.missiles(j).missileIdx;plot3(ax,[p.missileInit(m,1),0],[p.missileInit(m,2),0],[p.missileInit(m,3),0],'--','Color',t.missile(m,:),'LineWidth',1.5,'DisplayName',char(p.missileIds(m)));end;for k=1:numel(b),q=b(k);c=t.uav(q.uavIdx,:);u=p.uavInit(q.uavIdx,:);plot3(ax,[u(1),q.releasePoint(1)],[u(2),q.releasePoint(2)],[u(3),q.releasePoint(3)],'-','Color',c,'LineWidth',1.7);plot3(ax,[q.releasePoint(1),q.burstPoint(1)],[q.releasePoint(2),q.burstPoint(2)],[q.releasePoint(3),q.burstPoint(3)],':','Color',c,'LineWidth',1.4);scatter3(ax,q.releasePoint(1),q.releasePoint(2),q.releasePoint(3),45,c,'o','filled');scatter3(ax,q.burstPoint(1),q.burstPoint(2),q.burstPoint(3),58,c,'^','filled');end;xlabel(ax,'X (m)');ylabel(ax,'Y (m)');zlabel(ax,'Z (m)');legend(ax,'Location','best');inline_style_q4(ax,t);end
function inline_intervals_q4(ax,e,t,titleText),hold(ax,'on');for j=1:numel(e.missiles),m=e.missiles(j);for k=1:size(m.intervals,1),rectangle(ax,'Position',[m.intervals(k,1),j-.28,diff(m.intervals(k,:)),.56],'FaceColor',t.missile(m.missileIdx,:),'EdgeColor','none');end;text(ax,max([m.intervals(:,2);0])+.2,j,sprintf('%.3f s',m.duration),'FontName',t.font);end;yticks(ax,1:numel(e.missiles));yticklabels(ax,arrayfun(@(m)sprintf('M%d',m.missileIdx),e.missiles,'UniformOutput',false));xlabel(ax,'时间 t (s)');ylabel(ax,'来袭导弹');title(ax,titleText);grid(ax,'on');inline_style_q4(ax,t);end
function inline_hist_q4(ax,h,label,c),plot(ax,h.Iteration,h.BestScore,'Color',c,'LineWidth',1.6,'DisplayName',label);end
function inline_style_q4(ax,t),set(ax,'FontName',t.font,'FontSize',10,'LineWidth',.8,'Box','on','Color',t.bg);end
function inline_export_q4(fig,file),exportgraphics(fig,file,'Resolution',300,'BackgroundColor','white');fprintf('[Q4绘图] 已保存：%s\n',file);end
