function runDir = create_run_directory(cfg, questionName)
%CREATE_RUN_DIRECTORY 创建带毫秒时间戳的本次运行目录。

stamp = char(datetime('now', 'Format', 'yyyyMMdd''T''HHmmssSSS'));
runDir = fullfile(cfg.outputRoot, questionName, stamp);
if ~exist(runDir, 'dir')
    [ok, message] = mkdir(runDir);
    if ~ok
        error('SmokeOutput:DirectoryCreateFailed', '无法创建结果目录 %s：%s', runDir, message);
    end
end
end
