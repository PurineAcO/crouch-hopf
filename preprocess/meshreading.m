function closed_all = meshreading(meshfile)
%% 1. 读取网格文件头,确定层数n_total和每层点数s_total

fprintf('meshreading: 开始读取网格文件 %s\n', meshfile);
fid = fopen(meshfile, 'r');
if fid == -1
    error('无法打开网格文件: %s', meshfile);
end
header = fgetl(fid);dims = sscanf(header, '%d');
if numel(dims) ~= 2
    fclose(fid);
    error('网格文件头部格式错误, 应为两个整数(层数, 每圈点数): %s', header);
end
i_total = dims(1);j_total = dims(2);

%% 2. 读取全部节点坐标
node_data = textscan(fid, '%f %f');
fclose(fid);
x = node_data{1};y = node_data{2};meshcnt = numel(x);

if meshcnt == 0
    error('网格文件 %s 中没有节点数据', meshfile);
end

if meshcnt ~= i_total * j_total
    error('网格文件 %s 中节点数不匹配: 期望 %d (=%d*%d), 实际 %d', ...
        meshfile, i_total * j_total, i_total, j_total, meshcnt);
end

fprintf('成功读取 %d 个网格点 (层数 %d, 每圈 %d 点).\n', meshcnt, i_total, j_total);
    
%% 3. 检测网格是否成环,修复非成环网格

tol = 1e-10;
closed_all = true;  % 判断整体是否开环的bool
first_open = 0;     % 第一处开环点位
for i = 1 : i_total
    idx = (i - 1) * j_total + 1 : i * j_total;
    if abs(x(idx(1)) - x(idx(end))) > tol || abs(y(idx(1)) - y(idx(end))) > tol
        closed_all = false;
        first_open = i;
        break;
    end
end

if closed_all % 所有圈首尾重合, 已成环, 无需补环
    s = j_total - 1;
    fprintf('检测结果: 所有 %d 圈均已成环(首尾重合), 无需补环.\n', i_total);
else     % 有圈不成环: 将每圈第一个点复制到圈尾补成环, 并重新输出节点数据
    s = j_total;
    x_new = zeros(j_total + 1, i_total);
    y_new = zeros(j_total + 1, i_total);
    for i = 1 : i_total
        idx = (i - 1) * j_total + 1 : i * j_total;
        x_new(1 : j_total, i) = x(idx);
        y_new(1 : j_total, i) = y(idx);
        x_new(j_total + 1, i) = x(idx(1));   
        y_new(j_total + 1, i) = y(idx(1));
    end
    outfile = fullfile(script_dir, 'meshdata_corrected.txt');
    fid_out = fopen(outfile, 'w');
    if fid_out == -1
        error('无法写入节点数据文件: %s', outfile);
    end
    fprintf(fid_out, '%d %d\n', i_total, j_total + 1);
    for i = 1 : i_total
        for j = 1 : j_total + 1
            fprintf(fid_out, '%.16f %.16f\n', x_new(j, i), y_new(j, i));
        end
    end
    fclose(fid_out);

    fprintf('检测结果: 第 %d 圈首尾不重合, 已将所有圈补成环, 并输出节点数据到 %s.\n', ...
        first_open, outfile);
end

%% 4. 结果输出

    fprintf('\n========== 网格统计 ==========\n');
    fprintf('层数 n             : %d\n', i_total);
    fprintf('每圈节点数 s(不含回绕点): %d\n', s);
    fprintf('==============================\n');

end