% pro-process 进行RANS数据的前处理.

%% 第一部分
script_dir = fileparts(mfilename('fullpath'));
% diary 覆写模式: 先删除旧日志再开启
preprocess_log = fullfile(script_dir, 'preprocess.log');
if exist(preprocess_log, 'file')
    delete(preprocess_log);
end
diary(preprocess_log);
fprintf("当前目录为%s\n",script_dir);
fprintf("========================================================\n");

%% pre-process 自主编程区

% ---- 1. 读取网格 ----
meshfile = fullfile(script_dir, '..', 'testdata', 'yuanzhudata.txt');
[n_total, s_total, node_x, node_y] = meshreading(meshfile);
fprintf("========================================================\n");

% ---- 2. 几何形成 ----
resultfile = fullfile(script_dir, '..', 'testdata', 'result.csv');
geometry(n_total, s_total, node_x, node_y, resultfile);
fprintf("========================================================\n");

% ---- 3. 可视化网格、边法向量与物理量云图 ----
mesh_visual();
fprintf("========================================================\n");

%% 第三部分
fprintf('preprocess 完成: 层数 n=%d, 每圈节点数 s=%d (不含回绕点).\n', n_total, s_total);
diary off;