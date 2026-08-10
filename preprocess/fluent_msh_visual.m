function fluent_msh_visual()
% fluent_msh_visual  可视化重构后的 C-block 网格(读取 icm_cblock.txt).
%
% 读取 fluent_msh_geometry 导出的 icm_cblock.txt, 绘制:
%   图1: 全网格单元按块着色(半O块 / 矩形块);
%   图2: 结构化网格线(按 (i,j) 索引抽样), 展示两块结构;
%   图3: 单元中心;
%   图4: 机翼附近放大(前缘包裹区).
% 当场显示, 不保存图片.

script_dir = fileparts(mfilename('fullpath'));
resfile = fullfile(script_dir, 'icm_cblock.txt');

%% ---- 读取结果 ----
fid = fopen(resfile, 'r');
if fid < 0
    error('请先运行 fluent_msh_geometry 生成 %s', resfile);
end
% 跳到数据头
while true
    line = fgetl(fid);
    if ~ischar(line); error('结果文件为空'); end
    if ~isempty(line) && line(1) ~= '#'
        dims = sscanf(line, '%d %d');
        break;
    end
end
N = dims(1); n_node = dims(2);

blk = zeros(N, 1); ci = zeros(N, 1); cj = zeros(N, 1);
cx = zeros(N, 1); cy = zeros(N, 1);
cn = zeros(N, 4); nb = zeros(N, 4); bt = zeros(N, 4);
c = 0;
while c < N
    line = fgetl(fid);
    if ~ischar(line); error('结果文件不完整'); end
    if isempty(line) || line(1) == '#'; continue; end
    row = sscanf(line, '%f');
    c = c + 1;
    blk(c) = row(1); ci(c) = row(2); cj(c) = row(3);
    cx(c) = row(4); cy(c) = row(5);
    cn(c, :) = row(7 : 10); nb(c, :) = row(11 : 14); bt(c, :) = row(15 : 18);
end
% 节点表
x = zeros(1, n_node); y = zeros(1, n_node);
while true
    line = fgetl(fid);
    if ~ischar(line); break; end
    if isempty(line) || line(1) == '#'; continue; end
    row = sscanf(line, '%f');
    if numel(row) >= 3
        x(row(1)) = row(2); y(row(1)) = row(3);
    end
end
fclose(fid);

n1 = sum(blk == 1); n2 = sum(blk == 2);
fprintf('读取 %d 单元 (半O %d + 矩形 %d), %d 节点\n', N, n1, n2, n_node);

%% ---- 绘图辅助 ----
patch_cell = @(ax, c, fc) patch(ax, ...
    [x(cn(c, 1)) x(cn(c, 2)) x(cn(c, 3)) x(cn(c, 4)) x(cn(c, 1))], ...
    [y(cn(c, 1)) y(cn(c, 2)) y(cn(c, 3)) y(cn(c, 4)) y(cn(c, 1))], ...
    fc, 'EdgeColor', 'none');

%% ---- 图1: 按块着色 ----
fig = figure('Position', [80 80 1000 900]);
ax = axes('Parent', fig); hold(ax, 'on');
for c = 1 : N
    if blk(c) == 1
        patch_cell(ax, c, [0.55 0.75 0.95]);
    else
        patch_cell(ax, c, [0.95 0.72 0.55]);
    end
end
xlabel('X (m)'); ylabel('Y (m)');
title(sprintf('C-block: 半O块 %d 单元(蓝) + 矩形块 %d 单元(橙)', n1, n2));
axis equal; grid on; hold off;

%% ---- 图2: 结构化网格线 (按 (i,j) 抽样, 支数现场读取) ----
fig = figure('Position', [90 90 1000 900]);
ax = axes('Parent', fig); hold(ax, 'on');
% 半O: 按 j 抽周向线
for j = 1 : 20 : max(cj(blk == 1))
    sel = blk == 1 & cj == j;
    if any(sel)
        plot(ax, cx(sel), cy(sel), '-', 'Color', [0.2 0.4 0.7], 'LineWidth', 0.6);
    end
end
% 矩形: 按 j 抽展向线
for j = 1 : 20 : max(cj(blk == 2))
    sel = blk == 2 & cj == j;
    if any(sel)
        plot(ax, cx(sel), cy(sel), '-', 'Color', [0.8 0.45 0.15], 'LineWidth', 0.6);
    end
end
xlabel('X (m)'); ylabel('Y (m)');
title('结构化网格线(抽样): 蓝=半O (i,j), 橙=矩形 (i,j)');
axis equal; grid on; hold off;

%% ---- 图3: 单元中心 ----
fig = figure('Position', [100 100 1000 900]);
ax = axes('Parent', fig); hold(ax, 'on');
h1 = plot(ax, cx(blk == 1), cy(blk == 1), '.', 'Color', [0.2 0.4 0.7], 'MarkerSize', 3);
h2 = plot(ax, cx(blk == 2), cy(blk == 2), '.', 'Color', [0.85 0.4 0.1], 'MarkerSize', 3);
xlabel('X (m)'); ylabel('Y (m)');
legend(ax, [h1 h2], {'半O单元中心', '矩形单元中心'}, 'Location', 'best');
title(sprintf('单元中心 (%d 个)', N));
axis equal; grid on; hold off;

%% ---- 图4: 机翼附近放大 ----
fig = figure('Position', [110 110 1000 900]);
ax = axes('Parent', fig); hold(ax, 'on');
for c = 1 : N
    if blk(c) == 1
        patch_cell(ax, c, [0.55 0.75 0.95]);
    else
        patch_cell(ax, c, [0.95 0.72 0.55]);
    end
end
xlim(ax, [-20 30]); ylim(ax, [-20 20]);
xlabel('X (m)'); ylabel('Y (m)');
title('机翼附近放大 (C-block 前缘包裹区)');
axis equal; grid on; hold off;
end