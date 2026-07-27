function fig = animate_plan_3d(problem,plan,evaluation,cfg,questionName)
%ANIMATE_PLAN_3D 在 MATLAB 中实时显示导弹、无人机、弹体和烟幕云团运动。

bombs=derive_bombs(plan,problem);
missileIds=[evaluation.missiles.missileIdx];
uavIds=unique([plan.uavIdx]);
palette=[0,114,178;213,94,0;0,158,115;204,121,167;230,159,0;86,180,233]/255;
fig=figure('Name',[questionName,' - 3D dynamic simulation'],'NumberTitle','off', ...
    'Color','w','Position',[120,80,1250,760],'Visible','on');
ax=axes(fig);hold(ax,'on');grid(ax,'on');view(ax,34,24);axis(ax,'vis3d');
[xc,yc,zc]=cylinder(problem.target.radius,32);zc=zc.*problem.target.height;
surf(ax,xc+problem.target.center(1),yc+problem.target.center(2),zc+problem.target.center(3), ...
    'FaceColor',[0.2,0.2,0.2],'FaceAlpha',0.55,'EdgeColor','none');
scatter3(ax,0,0,0,75,'k','x','LineWidth',2);
xlabel(ax,'X (m)');ylabel(ax,'Y (m)');zlabel(ax,'Z (m)');
set(ax,'FontName','Arial','FontSize',10,'LineWidth',0.8,'Color',[0.985,0.985,0.985]);
xlim(ax,[-500,21000]);ylim(ax,[-4000,4000]);zlim(ax,[0,2600]);

missileMarker=gobjects(1,numel(missileIds));missileTrail=gobjects(1,numel(missileIds));losLine=gobjects(1,numel(missileIds));
missileX=cell(1,numel(missileIds));missileY=cell(1,numel(missileIds));missileZ=cell(1,numel(missileIds));
for j=1:numel(missileIds)
    color=palette(j,:);
    missileMarker(j)=scatter3(ax,NaN,NaN,NaN,65,color,'>','filled','DisplayName',sprintf('M%d',missileIds(j)));
    missileTrail(j)=plot3(ax,NaN,NaN,NaN,'--','Color',color,'LineWidth',1.3,'HandleVisibility','off');
    losLine(j)=plot3(ax,NaN,NaN,NaN,'-','Color',[0.75,0.2,0.2],'LineWidth',1.2,'HandleVisibility','off');
    missileX{j}=[];missileY{j}=[];missileZ{j}=[];
end
uavMarker=gobjects(1,numel(uavIds));uavTrail=gobjects(1,numel(uavIds));
uavX=cell(1,numel(uavIds));uavY=cell(1,numel(uavIds));uavZ=cell(1,numel(uavIds));
for u=1:numel(uavIds)
    color=palette(mod(u+2-1,size(palette,1))+1,:);
    uavMarker(u)=scatter3(ax,NaN,NaN,NaN,55,color,'s','filled','DisplayName',char(problem.uavIds(uavIds(u))));
    uavTrail(u)=plot3(ax,NaN,NaN,NaN,'-','Color',color,'LineWidth',1.4,'HandleVisibility','off');
    uavX{u}=[];uavY{u}=[];uavZ{u}=[];
end
bombMarker=gobjects(1,numel(bombs));cloudSurface=gobjects(1,numel(bombs));
[sx,sy,sz]=sphere(14);
for k=1:numel(bombs)
    color=palette(mod(k-1,size(palette,1))+1,:);
    bombMarker(k)=scatter3(ax,NaN,NaN,NaN,34,color,'o','filled','Visible','off','HandleVisibility','off');
    cloudSurface(k)=surf(ax,sx,sy,sz,'FaceColor',color,'FaceAlpha',0.20, ...
        'EdgeColor','none','Visible','off','HandleVisibility','off');
end
statusText=text(ax,0.015,0.97,'','Units','normalized','VerticalAlignment','top', ...
    'FontName','Arial','FontSize',11,'FontWeight','bold','BackgroundColor','w','Margin',5);
legend(ax,'Location','northeastoutside');
targetPoint=problem.target.center+[0,0,problem.target.height/2];
sample=create_target_samples(problem.target,cfg.fastSample);
tEnd=max(problem.hitTime(missileIds));
timeGrid=0:cfg.animationTimeStep:tEnd;
for tt=1:numel(timeGrid)
    if ~isgraphics(fig),break;end
    t=timeGrid(tt);
    statusStrings=strings(1,numel(missileIds));
    for j=1:numel(missileIds)
        idx=missileIds(j); currentTime=min(t,problem.hitTime(idx));
        p=missile_position(problem,idx,currentTime);
        missileX{j}(end+1)=p(1);missileY{j}(end+1)=p(2);missileZ{j}(end+1)=p(3);
        set(missileMarker(j),'XData',p(1),'YData',p(2),'ZData',p(3));
        set(missileTrail(j),'XData',missileX{j},'YData',missileY{j},'ZData',missileZ{j});
        [margin,~,~]=coverage_margin(currentTime,idx,bombs,problem,sample,cfg.geometryTolerance);
        covered=margin<=0;
        if covered,losColor=[0,0.55,0.25];state='SHIELDED';else,losColor=[0.80,0.20,0.16];state='VISIBLE';end
        set(losLine(j),'XData',[p(1),targetPoint(1)],'YData',[p(2),targetPoint(2)], ...
            'ZData',[p(3),targetPoint(3)],'Color',losColor);
        statusStrings(j)=sprintf('M%d: %s',idx,state);
    end
    for u=1:numel(uavIds)
        idx=find([plan.uavIdx]==uavIds(u),1);
        direction=[cos(plan(idx).theta),sin(plan(idx).theta),0];
        p=problem.uavInit(uavIds(u),:)+plan(idx).speed*t*direction;
        uavX{u}(end+1)=p(1);uavY{u}(end+1)=p(2);uavZ{u}(end+1)=p(3);
        set(uavMarker(u),'XData',p(1),'YData',p(2),'ZData',p(3));
        set(uavTrail(u),'XData',uavX{u},'YData',uavY{u},'ZData',uavZ{u});
    end
    for k=1:numel(bombs)
        if t<bombs(k).releaseTime
            set(bombMarker(k),'Visible','off');set(cloudSurface(k),'Visible','off');
        elseif t<bombs(k).burstTime
            s=t-bombs(k).releaseTime;
            p=bombs(k).releasePoint+bombs(k).speed*s*bombs(k).direction-0.5*problem.gravity*s^2*[0,0,1];
            set(bombMarker(k),'XData',p(1),'YData',p(2),'ZData',p(3),'Visible','on');
            set(cloudSurface(k),'Visible','off');
        elseif t<=bombs(k).cloudEndTime
            c=bombs(k).burstPoint-problem.smokeSinkSpeed*(t-bombs(k).burstTime)*[0,0,1];
            set(bombMarker(k),'Visible','off');
            set(cloudSurface(k),'XData',problem.smokeRadius*sx+c(1), ...
                'YData',problem.smokeRadius*sy+c(2),'ZData',problem.smokeRadius*sz+c(3),'Visible','on');
        else
            set(bombMarker(k),'Visible','off');set(cloudSurface(k),'Visible','off');
        end
    end
    set(statusText,'String',sprintf('t = %.2f s\n%s',t,strjoin(statusStrings,' | ')));
    title(ax,sprintf('%s dynamic simulation',questionName),'FontName','Arial','FontWeight','bold');
    drawnow limitrate;
    if cfg.animationPause>0,pause(cfg.animationPause);end
end
rotate3d(fig,'on');
end
