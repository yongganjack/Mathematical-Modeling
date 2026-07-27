function [psoOptions, deOptions] = build_solver_options(cfg, seedOffset)
%BUILD_SOLVER_OPTIONS 将统一配置转换为自定义 PSO/DE 选项。

if nargin < 2
    seedOffset = 0;
end
psoOptions = cfg.pso;
psoOptions.seed = cfg.masterSeed + seedOffset;
psoOptions.useParallel = cfg.useParallel;
psoOptions.debug = cfg.debug;
psoOptions.functionTolerance = 1e-8;

deOptions = cfg.de;
deOptions.seed = cfg.masterSeed + 1000 + seedOffset;
deOptions.useParallel = cfg.useParallel;
deOptions.debug = cfg.debug;
deOptions.functionTolerance = 1e-8;
end
