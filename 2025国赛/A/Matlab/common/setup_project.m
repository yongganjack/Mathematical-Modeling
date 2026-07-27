function projectRoot = setup_project()
%SETUP_PROJECT 将项目公共目录和五个问题目录加入 MATLAB 路径。

thisFile = mfilename('fullpath');
projectRoot = fileparts(fileparts(thisFile));
addpath(genpath(projectRoot));
end
