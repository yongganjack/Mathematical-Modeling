function cfg = quick_config()
%QUICK_CONFIG 开发与冒烟运行配置，预算较小，不用于论文最终值。

cfg.name = 'quick';
cfg.masterSeed = 2025;
cfg.projectRoot = fileparts(fileparts(mfilename('fullpath')));
cfg.outputRoot = fullfile(cfg.projectRoot, 'outputs');
cfg.enablePlots = true;
cfg.showStaticFigures = true;
cfg.livePlots = true;
cfg.saveFigures = false;
cfg.animationTimeStep = 0.30;
cfg.animationPause = 0.015;
cfg.useParallel = false;
cfg.debug = true;

cfg.fastSample.sideAngles = 18;
cfg.fastSample.sideHeights = 5;
cfg.fastSample.topAngles = 18;
cfg.fastSample.topRadii = 3;
cfg.verifySample.sideAngles = 48;
cfg.verifySample.sideHeights = 9;
cfg.verifySample.topAngles = 48;
cfg.verifySample.topRadii = 6;

cfg.fastTimeStep = 0.25;
cfg.verifyTimeStep = 0.05;
cfg.rootTolerance = 1.0e-5;
cfg.intervalMergeTolerance = 2.0e-4;
cfg.geometryTolerance = 1.0e-9;
cfg.coverageWeight = 1.0e-4;
cfg.proximityWeight = 1.0e-4;
cfg.lexicographicTolerance = 0.02;

cfg.pso.swarmSize = 18;
cfg.pso.maxIterations = 18;
cfg.pso.maxStallIterations = 6;
cfg.pso.inertiaMax = 0.90;
cfg.pso.inertiaMin = 0.35;
cfg.pso.cognitiveWeight = 1.45;
cfg.pso.socialWeight = 1.55;
cfg.pso.velocityFraction = 0.20;
cfg.pso.restartFraction = 0.25;

cfg.de.populationSize = 20;
cfg.de.maxIterations = 18;
cfg.de.memorySize = 5;
cfg.de.archiveRate = 1.4;
cfg.de.minPopulationSize = 8;
cfg.de.maxStallIterations = 7;

cfg.q2.keepCandidates = 6;
cfg.q3.keepCandidates = 5;
cfg.q4.keepCandidates = 5;
cfg.q4.seedPerUav = 8;
cfg.q4.initialCombinationCount = 36;
cfg.q4.blockCycles = 1;
cfg.q4.blockIterations = 8;
cfg.q4.blockPopulation = 12;
cfg.q5.routeIterations = 6;
cfg.q5.mixedRouteIterations = 4;
cfg.q5.routePopulation = 10;
cfg.q5.routesPerUavMissile = 1;
cfg.q5.seedPerPattern = 12;
cfg.q5.selectionSeedCount = 120;
cfg.q5.selectionIterations = 15;
cfg.q5.selectionSwarmSize = 20;
cfg.q5.continuousRefineIterations = 4;
cfg.q5.maxRoutesPerUav = 40;
end
