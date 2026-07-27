function validation = export_result_workbook(questionNumber, templatePath, outputPath, plan, problem, cfg)
%EXPORT_RESULT_WORKBOOK 复制官方模板、写入三位小数结果并执行回读校验。

if ~isfile(templatePath)
    error('SmokeOutput:MissingTemplate', '未找到 Excel 模板：%s', templatePath);
end
outputFolder = fileparts(outputPath);
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end
[ok, message] = copyfile(templatePath, outputPath, 'f');
if ~ok
    error('SmokeOutput:TemplateCopyFailed', '复制 Excel 模板失败：%s', message);
end

bombs = derive_bombs(plan, problem);
contributions = compute_sequential_contributions(plan, problem, cfg);

switch questionNumber
    case 3
        [~, order] = sort([bombs.bombId]);
        bombs = bombs(order);
        contributions = contributions(order);
        rows = cell(3,10);
        for k = 1:3
            rows(k,:) = make_q3_row(bombs(k), contributions(k));
        end
        writecell(rows, outputPath, 'Sheet', 1, 'Range', 'A2:J4');
        expectedSize = [6,10];
    case 4
        [~, order] = sort([bombs.uavIdx]);
        bombs = bombs(order);
        contributions = contributions(order);
        rows = cell(3,10);
        for k = 1:3
            rows(k,:) = make_q4_row(bombs(k), contributions(k), problem);
        end
        writecell(rows, outputPath, 'Sheet', 1, 'Range', 'A2:J4');
        expectedSize = [6,10];
    case 5
        rows = cell(15,12);
        for u = 1:5
            for b = 1:3
                row = (u-1)*3+b;
                rows{row,1} = char(problem.uavIds(u));
                rows{row,4} = b;
            end
        end
        for k = 1:numel(bombs)
            row = (bombs(k).uavIdx-1)*3 + bombs(k).bombId;
            rows(row,:) = make_q5_row(bombs(k), contributions(k), problem);
        end
        writecell(rows, outputPath, 'Sheet', 1, 'Range', 'A2:L16');
        expectedSize = [18,12];
    otherwise
        error('SmokeOutput:InvalidQuestionNumber', '仅问题3、4、5需要输出官方 Excel 模板。');
end

readBack = readcell(outputPath, 'Sheet', 1);
validation.outputPath = outputPath;
validation.fileExists = isfile(outputPath);
validation.actualSize = size(readBack);
validation.expectedMinimumSize = expectedSize;
validation.sizeValid = size(readBack,2) == expectedSize(2) && size(readBack,1) >= expectedSize(1)-2;
validation.numericPrecision = 3;
validation.passed = validation.fileExists && validation.sizeValid;
if ~validation.passed
    error('SmokeOutput:ExcelReadbackFailed', '结果文件写入后结构校验未通过：%s', outputPath);
end
end

function row = make_q3_row(bomb, contribution)
row = {heading_degrees(bomb.theta), round(bomb.speed,3), bomb.bombId, ...
    round(bomb.releasePoint(1),3), round(bomb.releasePoint(2),3), round(bomb.releasePoint(3),3), ...
    round(bomb.burstPoint(1),3), round(bomb.burstPoint(2),3), round(bomb.burstPoint(3),3), ...
    round(contribution,3)};
end

function row = make_q4_row(bomb, contribution, problem)
row = {char(problem.uavIds(bomb.uavIdx)), heading_degrees(bomb.theta), round(bomb.speed,3), ...
    round(bomb.releasePoint(1),3), round(bomb.releasePoint(2),3), round(bomb.releasePoint(3),3), ...
    round(bomb.burstPoint(1),3), round(bomb.burstPoint(2),3), round(bomb.burstPoint(3),3), ...
    round(contribution,3)};
end

function row = make_q5_row(bomb, contribution, problem)
row = {char(problem.uavIds(bomb.uavIdx)), heading_degrees(bomb.theta), round(bomb.speed,3), ...
    bomb.bombId, round(bomb.releasePoint(1),3), round(bomb.releasePoint(2),3), ...
    round(bomb.releasePoint(3),3), round(bomb.burstPoint(1),3), ...
    round(bomb.burstPoint(2),3), round(bomb.burstPoint(3),3), ...
    round(contribution,3), char(problem.missileIds(bomb.targetId))};
end

function degrees = heading_degrees(theta)
degrees = round(mod(rad2deg(theta), 360), 3);
end
