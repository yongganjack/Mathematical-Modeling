%RUN_ALL 在 MATLAB 中直接点击 Run，依次执行问题1至问题5。
% quick 用于开发和流程检查；competition 用于正式高预算搜索。

clear; clc;
projectRoot = resolve_project_root_for_entry();
addpath(genpath(projectRoot));
defaultConfig = 'quick'; % 可直接改为 'competition'
continueOnError = true;

if strcmpi(defaultConfig,'competition')
    cfg = competition_config();
else
    cfg = quick_config();
end

solvers = {@run_question1,@run_question2,@run_question3,@run_question4,@run_question5};
for questionIdx = 1:numel(solvers)
    try
        solvers{questionIdx}(cfg);
    catch ME
        fprintf(2,'[批量运行异常] 问题%d执行失败：%s\n',questionIdx,ME.message);
        if ~continueOnError
            rethrow(ME);
        end
    end
end
fprintf('\n全部问题的批量调用已结束，请查看 outputs 目录中的独立运行结果。\n');

function projectRoot = resolve_project_root_for_entry()
entryFile = mfilename('fullpath');
if isempty(entryFile)
    try, entryFile = matlab.desktop.editor.getActiveFilename; catch, entryFile = ''; end
end
if isempty(entryFile), entryFile = fullfile(pwd, 'run_all.m'); end
folder = fileparts(entryFile);
while true
    if isfolder(fullfile(folder, 'configs'))
        projectRoot = folder;
        return;
    end
    parent = fileparts(folder);
    if strcmp(parent, folder), break; end
    folder = parent;
end
error('SmokeProject:RootNotFound', '未找到包含 configs 文件夹的 MATLAB 项目根目录。');
end
