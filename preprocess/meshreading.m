function [n_total, s_total, node_x, node_y] = meshreading(meshfile)

script_dir = fileparts(mfilename('fullpath'));

fprintf('meshreading: 读取网格文件 %s\n', meshfile);

fid = fopen(meshfile, 'r');
if fid == -1
    error('无法打开网格文件: %s', meshfile);
end

header = fgetl(fid);
dims = sscanf(header, '%d');
if numel(dims) ~= 2
    fclose(fid);
    error('文件头应为两个整数(层数, 每圈点数)');
end
n_total = dims(1);
s_wrap  = dims(2);

node_data = textscan(fid, '%f %f');
fclose(fid);
x = node_data{1};
y = node_data{2};
meshcnt = numel(x);

if meshcnt == 0
    error('网格文件 %s 中没有节点数据', meshfile);
end
if meshcnt ~= n_total * s_wrap
    error('节点数不匹配: 期望 %d, 实际 %d', n_total * s_wrap, meshcnt);
end

fprintf('成功读取 %d 个网格点 (层数 %d, 每圈 %d 点).\n', meshcnt, n_total, s_wrap);

% 检测每圈是否成环(首点与末点重合)
tol = 1e-10;
closed_all = true;
for n = 1 : n_total
    idx = (n - 1) * s_wrap + 1 : n * s_wrap;
    if abs(x(idx(1)) - x(idx(end))) > tol || abs(y(idx(1)) - y(idx(end))) > tol
        closed_all = false;
        break;
    end
end

if closed_all
    s_total = s_wrap - 1;
    fprintf('所有 %d 圈均已成环, 无需补环.\n', n_total);
else
    s_total = s_wrap;
    fprintf('存在不成环的圈, 已补成环.\n');
end

% 节点矩阵(每圈 s_total+1 个点, 末列为回绕点)
node_x = zeros(n_total, s_total + 1);
node_y = zeros(n_total, s_total + 1);
for n = 1 : n_total
    idx = (n - 1) * s_wrap + 1 : n * s_wrap;
    node_x(n, 1 : s_wrap) = x(idx);
    node_y(n, 1 : s_wrap) = y(idx);
    if ~closed_all
        node_x(n, s_wrap + 1) = x(idx(1));
        node_y(n, s_wrap + 1) = y(idx(1));
    end
end

% 输出 mesh_last.txt
outfile = fullfile(script_dir, 'mesh_last.txt');
fid_out = fopen(outfile, 'w');
if fid_out == -1
    error('无法写入节点文件: %s', outfile);
end
fprintf(fid_out, '%d %d\n', n_total, s_total);
for n = 1 : n_total
    for s = 1 : s_total + 1
        fprintf(fid_out, '%.16f %.16f\n', node_x(n, s), node_y(n, s));
    end
end
fclose(fid_out);
fprintf('节点文件已输出到 %s (每圈 %d 个点, 含回绕点).\n', outfile, s_total + 1);

fprintf('网格统计: 层数 n=%d, 每圈节点数 s=%d (不含回绕点).\n', n_total, s_total);
end