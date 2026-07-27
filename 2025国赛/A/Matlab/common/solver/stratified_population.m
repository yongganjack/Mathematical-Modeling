function population = stratified_population(populationSize, lowerBound, upperBound)
%STRATIFIED_POPULATION 生成逐维分层随机初始种群，减少聚集和空白区域。

dimension = numel(lowerBound);
population = zeros(populationSize, dimension);
for d = 1:dimension
    strata = ((0:(populationSize-1))' + rand(populationSize, 1)) / populationSize;
    strata = strata(randperm(populationSize));
    population(:, d) = lowerBound(d) + strata .* (upperBound(d) - lowerBound(d));
end
end
