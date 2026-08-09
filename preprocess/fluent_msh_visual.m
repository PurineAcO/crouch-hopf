function msh_visual(meshfile)
% 可视化 Fluent 网格文件 ICM.txt 中的节点坐标.
% 需要提供已经实现了翻译的文件.函数进行节点坐标的可视化.

%% 1. 寻找节点端索引行开始位置,依据.msh规范,应为(10(0 1 .... 0 2/3))
fid = fopen(meshfile, 'r');
if fid < 0
    error('无法打开文件: %s', meshfile);
end

while true
    line = fgetl(fid);
    if ~ischar(line)
        error('未找到节点段索引行 (10 (...)');
    end
    if strncmp(strtrim(line), '(10 (', 5)
        break;
    end
end

% 下面第一个不是注释的行就是开始的节点位置,直到最后一个注释行为结束标
while true
    line = fgetl(fid);
    if ~ischar(line)
        error('未找到节点段数据头');
    end
    s = strtrim(line);
    if isempty(s)
        continue;
    end
    if s(1) == '(' && ~strncmp(s, '(0 "', 4) && ~strcmp(s(end-1:end), '))')
        break;
    end
end
tokens = regexp(line, '-?\d+', 'match');
nval = str2double(tokens{end});   % 每节点数据个数(本文件为 3: x y z)

%% 2. 读取节点数据
rows = {};
while true
    line = fgetl(fid);
    if ~ischar(line)
        error('节点段缺少结束 ")" 行');
    end
    s = strtrim(line);
    if strcmp(s, ')')
        break;   % 数据段结束
    end
    if isempty(s) || s(1) == '('
        continue;   % 跳过空行与块分隔括号行
    end
    rows{end + 1} = sscanf(line, '%f').'; %#ok<AGROW>
end
fclose(fid);

if isempty(rows)
    error('未读取到任何节点数据');
end
A = vertcat(rows{:});        % N x (1+nval): [id, x, y, z, ...]
n_node = size(A, 1);

% 提取前两维坐标(兼容 2 列/3 列两种节点格式)
if size(A, 2) >= 3
    x = A(:, 2); y = A(:, 3);
else
    error('节点数据列数不足, 无法提取 x/y 坐标');
end

fprintf('读取到 %d 个节点, 每节点 %d 个数据.\n', n_node, nval);

%% 3. 可视化
figure('Position', [100 100 900 900]);
scatter(x, y, 2, 1 : n_node, 'filled');   % 按节点序号着色
colormap(parula);
colorbar;
xlabel('X (m)'); ylabel('Y (m)');
title(sprintf('Fluent mesh nodes (ICM.txt): %d nodes', n_node));
axis equal; grid on;
end

script_dir = fileparts(mfilename('fullpath'));
meshfile = fullfile(script_dir, '..', 'testdata', 'ICM.txt');
msh_visual(meshfile);