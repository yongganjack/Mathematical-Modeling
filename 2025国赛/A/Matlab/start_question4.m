function output = start_question4()
%START_QUESTION4 问题4无参数一键入口，直接点击 Run。
clear; clc;
output = start_question_local(4, 'quick');
end

function output = start_question_local(questionNumber, configName)
projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot,'configs'),'-begin');
addpath(genpath(fullfile(projectRoot,'common')),'-begin');
addpath(fullfile(projectRoot,sprintf('question%d',questionNumber)),'-begin');
rehash path;
if strcmpi(configName,'competition'), cfg=competition_config(); else, cfg=quick_config(); end
fprintf('[入口] 项目根目录：%s\n',projectRoot);
fprintf('[入口] 已加载配置：%s\n',fullfile(projectRoot,'configs',[cfg.name,'_config.m']));
output = run_question4(cfg);
end
