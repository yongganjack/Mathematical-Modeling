function cfg = competition_config()
%COMPETITION_CONFIG 正式搜索配置。运行时间较长，结果仍需收敛复核。

cfg = quick_config();
cfg.name = 'competition';
cfg.useParallel = true;
cfg.debug = true;

cfg.fastSample.sideAngles = 36;
cfg.fastSample.sideHeights = 8;
cfg.fastSample.topAngles = 36;
cfg.fastSample.topRadii = 5;
cfg.verifySample.sideAngles = 96;
cfg.verifySample.sideHeights = 15;
cfg.verifySample.topAngles = 96;
cfg.verifySample.topRadii = 10;
cfg.fastTimeStep = 0.10;
cfg.verifyTimeStep = 0.02;
cfg.rootTolerance = 1.0e-6;
cfg.intervalMergeTolerance = 5.0e-5;

cfg.pso.swarmSize = 72;
cfg.pso.maxIterations = 140;
cfg.pso.maxStallIterations = 28;
cfg.de.populationSize = 80;
cfg.de.maxIterations = 140;
cfg.de.minPopulationSize = 18;
cfg.de.maxStallIterations = 30;

cfg.q2.keepCandidates = 20;
cfg.q3.keepCandidates = 16;
cfg.q4.keepCandidates = 16;
cfg.q4.seedPerUav = 18;
cfg.q4.initialCombinationCount = 120;
cfg.q4.blockCycles = 2;
cfg.q4.blockIterations = 35;
cfg.q4.blockPopulation = 40;
cfg.q5.routeIterations = 55;
cfg.q5.mixedRouteIterations = 25;
cfg.q5.routePopulation = 36;
cfg.q5.routesPerUavMissile = 3;
cfg.q5.seedPerPattern = 30;
cfg.q5.selectionSeedCount = 500;
cfg.q5.selectionIterations = 120;
cfg.q5.selectionSwarmSize = 80;
cfg.q5.continuousRefineIterations = 45;
cfg.q5.maxRoutesPerUav = 100;
end
