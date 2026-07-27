function [bestPlan, bestEvaluation, verificationTable] = verify_candidate_set(candidateMatrix, decoder, problem, cfg, missileIds, keepCount)
%VERIFY_CANDIDATE_SET 对候选去重并用高精度评价器统一排序。

if isempty(candidateMatrix)
    error('SmokeOptimization:EmptyCandidateSet', '没有可供高精度验证的优化候选。');
end
candidateMatrix = unique(round(candidateMatrix, 10), 'rows', 'stable');
candidateMatrix = candidateMatrix(1:min(keepCount,size(candidateMatrix,1)),:);
scores = -inf(size(candidateMatrix,1),1);
durations = zeros(size(candidateMatrix,1),numel(missileIds));
plans = cell(size(candidateMatrix,1),1);
evaluations = cell(size(candidateMatrix,1),1);
for k = 1:size(candidateMatrix,1)
    plans{k} = decoder(candidateMatrix(k,:));
    evaluations{k} = evaluate_plan(plans{k}, problem, cfg, missileIds, 'verify');
    scores(k) = evaluations{k}.totalDuration;
    durations(k,:) = [evaluations{k}.missiles.duration];
    fprintf('[验证] 候选 %d/%d，高精度总遮蔽时长 %.6f s。\n', ...
        k, size(candidateMatrix,1), scores(k));
end
[~, bestIdx] = max(scores);
bestPlan = plans{bestIdx};
bestEvaluation = evaluations{bestIdx};
verificationTable = array2table([(1:numel(scores))', scores, durations], ...
    'VariableNames', [{'Candidate','TotalDuration'}, ...
    arrayfun(@(x) sprintf('M%dDuration',x), missileIds, 'UniformOutput',false)]);
end
