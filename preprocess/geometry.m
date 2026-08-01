function geometry(n_total, s_total, node_x, node_y, resultfile)

script_dir = fileparts(mfilename('fullpath'));
n_cell = n_total - 1;

fprintf('geometry: 开始几何形成 (n_total=%d, s_total=%d)\n', n_total, s_total);

%% 1. 单元体积与中心
cell_x  = zeros(n_cell, s_total);
cell_y  = zeros(n_cell, s_total);
cell_vol = zeros(n_cell, s_total);
for n = 1 : n_cell
    for s = 1 : s_total
        x1 = node_x(n, s);         y1 = node_y(n, s);
        x2 = node_x(n + 1, s);     y2 = node_y(n + 1, s);
        x3 = node_x(n + 1, s + 1); y3 = node_y(n + 1, s + 1);
        x4 = node_x(n, s + 1);     y4 = node_y(n, s + 1);

        c1 = x1 * y2 - x2 * y1;
        c2 = x2 * y3 - x3 * y2;
        c3 = x3 * y4 - x4 * y3;
        c4 = x4 * y1 - x1 * y4;
        signed_area = 0.5 * (c1 + c2 + c3 + c4);
        if abs(signed_area) > 1e-30
            cell_x(n, s) = ((x1 + x2) * c1 + (x2 + x3) * c2 + (x3 + x4) * c3 + (x4 + x1) * c4) / (6 * signed_area);
            cell_y(n, s) = ((y1 + y2) * c1 + (y2 + y3) * c2 + (y3 + y4) * c3 + (y4 + y1) * c4) / (6 * signed_area);
        end

        vec1 = [x3 - x1, y3 - y1];
        vec2 = [x2 - x4, y2 - y4];
        cell_vol(n, s) = 0.5 * abs(vec1(1) * vec2(2) - vec1(2) * vec2(1));
    end
end
fprintf('单元: %d 层 x %d 个/层 = %d 个\n', n_cell, s_total, n_cell * s_total);

%% 2. NS 面
ns_nx = zeros(n_total, s_total);
ns_ny = zeros(n_total, s_total);
ns_mx = zeros(n_total, s_total);
ns_my = zeros(n_total, s_total);
for n = 1 : n_total
    for s = 1 : s_total
        dx = node_x(n, s + 1) - node_x(n, s);
        dy = node_y(n, s + 1) - node_y(n, s);
        ns_nx(n, s) = dy;
        ns_ny(n, s) = -dx;
        ns_mx(n, s) = 0.5 * (node_x(n, s) + node_x(n, s + 1));
        ns_my(n, s) = 0.5 * (node_y(n, s) + node_y(n, s + 1));
    end
end
fprintf('NS 面: %d 层 x %d 个/层\n', n_total, s_total);

%% 3. WE 面
we_nx = zeros(n_cell, s_total + 1);
we_ny = zeros(n_cell, s_total + 1);
we_mx = zeros(n_cell, s_total + 1);
we_my = zeros(n_cell, s_total + 1);
for n = 1 : n_cell
    for s = 1 : s_total + 1
        dx = node_x(n + 1, s) - node_x(n, s);
        dy = node_y(n + 1, s) - node_y(n, s);
        we_nx(n, s) = -dy;
        we_ny(n, s) = dx;
        we_mx(n, s) = 0.5 * (node_x(n, s) + node_x(n + 1, s));
        we_my(n, s) = 0.5 * (node_y(n, s) + node_y(n + 1, s));
    end
end
fprintf('WE 面: %d 层 x %d 个/层 (每圈总数+1)\n', n_cell, s_total + 1);

%% 4. sad
wall_mx = ns_mx(1, :);
wall_my = ns_my(1, :);
cell_sad = zeros(n_cell, s_total);
for n = 1 : n_cell
    d2 = (cell_x(n, :)' - wall_mx).^2 + (cell_y(n, :)' - wall_my).^2;
    cell_sad(n, :) = sqrt(min(d2, [], 2))';
end
fprintf('sad 计算完成\n');

%% 5.校验单元中心
res = readmatrix(resultfile);
res_i = res(:, 1);
res_j = res(:, 2);
res_x = res(:, 3);
res_y = res(:, 4);

if size(res, 1) ~= n_cell * s_total
    error('result.csv 行数 %d 与单元总数 %d 不匹配', size(res, 1), n_cell * s_total);
end
exp_i = ceil((1 : n_cell * s_total)' / s_total);
exp_j = mod((1 : n_cell * s_total)' - 1, s_total) + 1;
if ~(isequal(res_i, exp_i) && isequal(res_j, exp_j))
    error('result.csv 未按 (i-1)*s_total+j 顺序排列');
end

tol = 1e-6;
n_bad = 0;
for n = 1 : n_cell
    for s = 1 : s_total
        k = (n - 1) * s_total + s;
        if abs(cell_x(n, s) - res_x(k)) > tol || abs(cell_y(n, s) - res_y(k)) > tol
            fprintf('不一致: 单元(%d,%d) 计算=(%.8e, %.8e), csv=(%.8e, %.8e), 差=(%.2e, %.2e)\n', ...
                n, s, cell_x(n, s), cell_y(n, s), res_x(k), res_y(k), ...
                abs(cell_x(n, s) - res_x(k)), abs(cell_y(n, s) - res_y(k)));
            n_bad = n_bad + 1;
        end
    end
end
if n_bad == 0
    fprintf('单元中心校验: 全部 %d 个单元与 result.csv 一致\n', n_cell * s_total);
else
    fprintf('单元中心校验: %d 个单元与 result.csv 不一致\n', n_bad);
end

%% 6. 输出 ransdata.txt
% 列: s n x y sad vol rho u v T miubl E_s E_n E_idx W_s W_n W_idx N_s N_n N_idx S_s S_n S_idx
% 边索引: E=(s+0.5,n), W=(s-0.5,n), N=(s,n+0.5), S=(s,n-0.5)
% 快速编号: WE 面 idx=(n-1)*(s_total+1)+s; NS 面 idx=(n-1)*s_total+s
outfile = fullfile(script_dir, 'ransdata.txt');
fid = fopen(outfile, 'w');
if fid == -1
    error('无法写入 %s', outfile);
end
fprintf(fid, 's n x y sad vol rho u v T miubl E_s E_n E_idx W_s W_n W_idx N_s N_n N_idx S_s S_n S_idx\n');
for n = 1 : n_cell
    for s = 1 : s_total
        k = (n - 1) * s_total + s;
        e_idx = (n - 1) * (s_total + 1) + (s + 1);
        w_idx = (n - 1) * (s_total + 1) + s;
        n_idx = n * s_total + s;
        s_idx = (n - 1) * s_total + s;
        fprintf(fid, '%d %d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.4f %d %d %.4f %d %d %.4f %d %d %.4f %d %d\n', ...
            s, n, cell_x(n, s), cell_y(n, s), cell_sad(n, s), cell_vol(n, s), ...
            res(k, 5), res(k, 8), res(k, 9), res(k, 7), res(k, 10), ...
            s + 0.5, n, e_idx, s - 0.5, n, w_idx, s, n + 0.5, n_idx, s, n - 0.5, s_idx);
    end
end
fclose(fid);
fprintf('已输出 %s\n', outfile);

%% 7. 输出 edge.txt 
% NS 面: NS s n idx c1_s c1_n c1_idx c2_s c2_n c2_idx nx ny mx my
% WE 面: WE s n idx c1_s c1_n c1_idx c2_s c2_n c2_idx nx ny mx my
% c1 低侧邻居(南/西), c2 高侧邻居(北/东), 边界外侧为 0 0 0
% 单元快速编号: Cell(n,s) -> (n-1)*s_total+s
outfile = fullfile(script_dir, 'edge.txt');
fid = fopen(outfile, 'w');
if fid == -1
    error('无法写入 %s', outfile);
end
fprintf(fid, 'type s n idx c1_s c1_n c1_idx c2_s c2_n c2_idx nx ny mx my\n');
for n = 1 : n_total
    for s = 1 : s_total
        idx = (n - 1) * s_total + s;
        if n == 1
            c1 = [0, 0, 0];
            c2 = [1, s, s];
        elseif n == n_total
            c1 = [n_total - 1, s, (n_total - 2) * s_total + s];
            c2 = [0, 0, 0];
        else
            c1 = [n - 1, s, (n - 2) * s_total + s];
            c2 = [n, s, (n - 1) * s_total + s];
        end
        fprintf(fid, 'NS %d %d %d %d %d %d %d %d %d %.10e %.10e %.10e %.10e\n', ...
            s, n, idx, c1(2), c1(1), c1(3), c2(2), c2(1), c2(3), ...
            ns_nx(n, s), ns_ny(n, s), ns_mx(n, s), ns_my(n, s));
    end
end
for n = 1 : n_cell
    for s = 1 : s_total + 1
        idx = (n - 1) * (s_total + 1) + s;
        if s == 1 || s == s_total + 1
            c1 = [n, s_total, (n - 1) * s_total + s_total];
            c2 = [n, 1, (n - 1) * s_total + 1];
        else
            c1 = [n, s - 1, (n - 1) * s_total + s - 1];
            c2 = [n, s, (n - 1) * s_total + s];
        end
        fprintf(fid, 'WE %d %d %d %d %d %d %d %d %d %.10e %.10e %.10e %.10e\n', ...
            s, n, idx, c1(2), c1(1), c1(3), c2(2), c2(1), c2(3), ...
            we_nx(n, s), we_ny(n, s), we_mx(n, s), we_my(n, s));
    end
end
fclose(fid);
fprintf('已输出 %s\n', outfile);

fprintf('geometry 完成\n');
end