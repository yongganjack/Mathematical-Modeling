function output = start_question1()
%START_QUESTION1 问题1无参数一键入口，直接点击 Run。
clear; clc;
output = start_question_local(1, 'quick');
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
output = run_question1(cfg);
end
