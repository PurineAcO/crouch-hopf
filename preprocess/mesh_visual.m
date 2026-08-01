function mesh_visual()

script_dir = fileparts(mfilename('fullpath'));

% ---- 读取节点(mesh_last.txt) ----
fid = fopen(fullfile(script_dir, 'mesh_last.txt'), 'r');
dims = sscanf(fgetl(fid), '%d');
n_total = dims(1); s_total = dims(2);
node = textscan(fid, '%f %f');
fclose(fid);
node_x = reshape(node{1}, s_total + 1, n_total)';
node_y = reshape(node{2}, s_total + 1, n_total)';

% ---- 读取边数据(edge.txt, 跳过表头) ----
fid = fopen(fullfile(script_dir, 'edge.txt'), 'r');
fgetl(fid);   % 表头
edge = textscan(fid, '%s %d %d %d %d %d %d %d %d %d %f %f %f %f');
fclose(fid);
etype = edge{1};
enx = edge{11}; eny = edge{12}; emx = edge{13}; emy = edge{14};
ns_mask = strcmp(etype, 'NS');
we_mask = strcmp(etype, 'WE');
ns_cnt = n_total * s_total;   % NS 面行数(WE 面行的偏移)

% ---- 域尺寸与箭头缩放 ----
outer = node_x(n_total, :);
outer_y = node_y(n_total, :);
domain_size = max(max(outer) - min(outer), max(outer_y) - min(outer_y));
arrow_scale = domain_size * 0.03;

% 抽样步长: 每个方向约 15 个箭头
skip_n = max(1, round(n_total / 15));
skip_s = max(1, round(s_total / 15));

%% 图1: 网格 + 边法向
fig = figure('Visible', 'off', 'Position', [100 100 900 900]);
hold on;

% 网格线(不进入图例)
for n = 1 : n_total
    plot(node_x(n, 1 : s_total + 1), node_y(n, 1 : s_total + 1), ...
        'Color', [0.27 0.51 0.71], 'LineWidth', 0.5, 'HandleVisibility', 'off');
end
for s = 1 : s_total
    plot(node_x(:, s), node_y(:, s), 'Color', [0.27 0.51 0.71], 'LineWidth', 0.5, ...
        'HandleVisibility', 'off');
end

% NS 面法向箭头(抽样)
sel = false(size(etype));
for n = 1 : skip_n : n_total
    for s = 1 : skip_s : s_total
        sel((n - 1) * s_total + s) = true;
    end
end
use = ns_mask & sel;
quiver(emx(use), emy(use), enx(use) * arrow_scale, eny(use) * arrow_scale, 0, ...
    'Color', [0 0.5451 0.5451], 'LineWidth', 1.2, 'MaxHeadSize', 0.8, ...
    'DisplayName', 'NS face normal');

% WE 面法向箭头(抽样)
sel = false(size(etype));
for n = 1 : skip_n : n_total - 1
    for s = 1 : skip_s : s_total + 1
        sel(ns_cnt + (n - 1) * (s_total + 1) + s) = true;
    end
end
use = we_mask & sel;
quiver(emx(use), emy(use), enx(use) * arrow_scale, eny(use) * arrow_scale, 0, ...
    'Color', [1 0.5490 0], 'LineWidth', 1.2, 'MaxHeadSize', 0.8, ...
    'DisplayName', 'WE face normal');

legend('Location', 'best');
xlabel('X'); ylabel('Y');
title(sprintf('O-block mesh: %d rings x %d points', n_total, s_total));
axis equal; grid on; hold off;

outfile = fullfile(script_dir, 'mesh_visual.png');
print(fig, outfile, '-dpng', '-r150');
fprintf('网格与边法向可视化已保存到 %s\n', outfile);
close(fig);

%% 图2: 物理量云图(rho u v T miubl, 定义在单元中心)
% 聚焦圆心附近: 显示范围取物面半径的 3 倍(远场近似自由来流, 对比度低)
cx0 = mean(node_x(1, 1 : s_total + 1));
cy0 = mean(node_y(1, 1 : s_total + 1));
R0 = max(hypot(node_x(1, 1 : s_total + 1) - cx0, node_y(1, 1 : s_total + 1) - cy0));
lim = 3 * R0;

rd = readmatrix(fullfile(script_dir, 'ransdata.txt'));
cx = rd(:, 3); cy = rd(:, 4);
fields = {rd(:, 7), rd(:, 8), rd(:, 9), rd(:, 10), rd(:, 11)};
names = {'rho', 'u', 'v', 'T', 'miubl'};

fig = figure('Visible', 'off', 'Position', [100 100 1500 900]);
for k = 1 : 5
    subplot(2, 3, k);
    scatter(cx, cy, 15, fields{k}, 'filled');
    colorbar; axis equal;
    xlim([cx0 - lim, cx0 + lim]);
    ylim([cy0 - lim, cy0 + lim]);
    title(names{k});
end
outfile = fullfile(script_dir, 'mesh_fields.png');
print(fig, outfile, '-dpng', '-r150');
fprintf('物理量云图已保存到 %s\n', outfile);
close(fig);
end