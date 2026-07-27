function sample = create_target_samples(target, sampleCfg)
%CREATE_TARGET_SAMPLES 生成圆柱侧面、顶面及轮廓边界采样点与外法向。

required = {'sideAngles','sideHeights','topAngles','topRadii'};
for k = 1:numel(required)
    if ~isfield(sampleCfg, required{k}) || sampleCfg.(required{k}) < 1
        error('SmokeModel:InvalidSampleConfig', '目标采样配置缺少字段 %s 或其值非法。', required{k});
    end
end

cx = target.center(1);
cy = target.center(2);
cz = target.center(3);
R = target.radius;
H = target.height;

phiSide = linspace(0, 2*pi, sampleCfg.sideAngles + 1);
phiSide(end) = [];
zSide = linspace(cz, cz + H, sampleCfg.sideHeights);
[PhiS, ZS] = ndgrid(phiSide, zSide);
sidePoints = [cx + R*cos(PhiS(:)), cy + R*sin(PhiS(:)), ZS(:)];
sideNormals = [cos(PhiS(:)), sin(PhiS(:)), zeros(numel(PhiS), 1)];

phiTop = linspace(0, 2*pi, sampleCfg.topAngles + 1);
phiTop(end) = [];
if sampleCfg.topRadii == 1
    radii = R;
else
    radii = linspace(R/sampleCfg.topRadii, R, sampleCfg.topRadii);
end
[PhiT, RT] = ndgrid(phiTop, radii);
topPoints = [cx + RT(:).*cos(PhiT(:)), cy + RT(:).*sin(PhiT(:)), ...
    repmat(cz + H, numel(PhiT), 1)];
topNormals = repmat([0, 0, 1], size(topPoints, 1), 1);

centerPoint = [cx, cy, cz + H];
sample.points = [sidePoints; topPoints; centerPoint];
sample.normals = [sideNormals; topNormals; 0, 0, 1];
sample.count = size(sample.points, 1);
end
