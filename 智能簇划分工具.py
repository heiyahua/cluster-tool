# 智能簇划分工具V7.py

import sys, os, base64, io, threading, time
import pandas as pd
import chardet
from scipy.spatial import KDTree
import numpy as np
from math import radians, cos, sin, asin, sqrt
import shutil
import openpyxl
from tqdm import tqdm

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QRadioButton,
    QProgressBar, QMessageBox, QButtonGroup, QGroupBox, QGridLayout, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QTextCursor, QIcon, QPixmap

class WorkerSignals(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal()
    error = pyqtSignal(str)

def detect_encoding(file_path):
    """
    检测文件编码格式，对中文编码做特殊处理以提高兼容性
    """
    with open(file_path, 'rb') as f:
        detected = chardet.detect(f.read(10000))
        encoding = detected['encoding']
        confidence = detected['confidence']
        
        print(f"🔍 检测到编码: {encoding}")
        
        # 对于中文编码，统一使用 gbk 以提高兼容性
        if encoding:
            encoding_lower = encoding.lower()
            if encoding_lower in ['gb2312', 'gbk', 'gb18030']:
                print("🔄 中文编码统一使用 gbk")
                return 'gbk'
        
        # 如果置信度较低，返回 None 让调用者选择默认编码
        if confidence < 0.7:
            print("⚠️ 检测置信度较低，建议使用默认编码")
            return None
            
        return encoding

def expand_by_cell_count(input_filename, temp_output_filename):
    try:
        encoding = detect_encoding(input_filename)
        df = pd.read_csv(input_filename, encoding=encoding)
        expanded_rows = []
        for _, row in df.iterrows():
            gNodeB_id = row['基站ID']
            cell_count = int(row['小区数量'])
            for i in range(1, cell_count + 1):
                new_row = row.to_dict()
                new_row['ECI'] = f'{gNodeB_id}-{i}'
                expanded_rows.append(new_row)
        result_df = pd.DataFrame(expanded_rows)
        result_df.to_csv(temp_output_filename, index=False, encoding='gbk')
        print(f"✅ 展开完成：{temp_output_filename}")
    except Exception as e:
        print(f"❌ 展开失败：{input_filename}，错误信息：{e}")
        raise

def split_csv_by_city(input_file, output_dir, chunk_size=300000, city_column='地市'):
    os.makedirs(output_dir, exist_ok=True)
    try:
        df = pd.read_csv(input_file, encoding='GBK', delimiter=',')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='GBK')

    df.columns = df.columns.str.strip()
    print(f"修正后的列名: {df.columns.tolist()}")

    if city_column not in df.columns:
        raise ValueError(f"CSV 文件中没有 '{city_column}' 这一列，请检查文件！")

    for city, group in df.groupby(city_column):
        group = group.reset_index(drop=True)
        file_count = 1
        for i in range(0, len(group), chunk_size):
            chunk = group.iloc[i:i + chunk_size]
            output_file = os.path.join(output_dir, f"{city}_{file_count}.csv")
            chunk.to_csv(output_file, index=False, encoding='GBK')
            file_count += 1

    print("📁 文件拆分完成！")

def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # 地球半径（单位：米）
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def cluster_city_files(input_folder, output_folder, city_column='地市', overwrite=False, neighbor_num=5000, cran_min_members=3, cell_num=24, site_num=8):    
    """
    对 input_folder 中每个城市文件执行聚类，并在原始数据中新增"簇编号"列，输出到 output_folder。
    使用KDTree提升聚类前的邻近点查找效率。
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith('.csv'):
            continue

        filepath = os.path.join(input_folder, filename)
        print(f"📂 聚类处理文件：{filename}")

        try:
            df = pd.read_csv(filepath, encoding='gbk')
        except Exception as e:
            print(f"❌ 读取失败：{filename}，错误信息：{e}")
            continue

        required_cols = ['基站ID', '经度', '纬度', 'ECI', 'C-RAN标识']
        if not all(col in df.columns for col in required_cols):
            print(f"⚠️ 跳过文件（缺字段）：{filename}")
            continue

        # 清洗 C-RAN标识
        df['C-RAN标识'] = df['C-RAN标识'].replace(['nan', 'NaN', '', None, pd.NA], np.nan)

        # 按城市、经纬度排序
        df = df.sort_values(by=[city_column, '经度', '纬度']).reset_index(drop=True)

        # 提取站点信息
        site_coords = df[['基站ID', '经度', '纬度']].drop_duplicates().set_index('基站ID')
        site_ecis = df.groupby('基站ID')['ECI'].apply(list).to_dict()
        site_cran = df.groupby('基站ID')['C-RAN标识'].first().to_dict()

        coords_array = site_coords[['经度', '纬度']].values
        bs_ids = site_coords.index.tolist()
        kdtree = KDTree(coords_array)

        # 构造邻居距离字典
        distance_dict = {}
        
        tqdm_file = sys.stdout if sys.stdout else open(os.devnull, 'w')

        for idx, bs_id in tqdm(
            enumerate(bs_ids),
            total=len(bs_ids),
            desc="计算基站距离",
            file=tqdm_file,
            ascii=True,
            mininterval=0.1
        ):
            k = min(len(bs_ids), neighbor_num)  # 限制最多查询5000个邻居
            dists, indices = kdtree.query(coords_array[idx], k=k)

            dists = np.atleast_1d(dists)
            indices = np.atleast_1d(indices)

            if len(indices) <= 1:
                continue

            neighbor_info = []
            for dist, neighbor_idx in zip(dists[1:], indices[1:]):  # 跳过自身
                neighbor_id = bs_ids[neighbor_idx]
                geo_dist = haversine(
                    coords_array[idx][0], coords_array[idx][1],
                    coords_array[neighbor_idx][0], coords_array[neighbor_idx][1]
                )
                neighbor_info.append((neighbor_id, geo_dist))

            distance_dict[bs_id] = sorted(neighbor_info, key=lambda x: x[1])

        # 聚类主过程
        visited, cluster_id = set(), 1
        unvisited_sites = set(site_coords.index)
        eci_to_cluster = {}

        # 优化聚类逻辑，减少重复代码
        def add_neighbors_to_cluster(cluster, ecis, candidate_neighbors, max_bs=site_num, max_eci=cell_num):
            """
            通用的添加邻居到簇的函数
            """
            for neighbor_id, _ in candidate_neighbors:
                if neighbor_id in visited:
                    continue
                
                tmp_cluster = cluster | {neighbor_id}
                tmp_ecis = ecis | set(site_ecis[neighbor_id])
                
                if len(tmp_cluster) <= max_bs and len(tmp_ecis) <= max_eci:
                    cluster.add(neighbor_id)
                    ecis.update(site_ecis[neighbor_id])
                    visited.add(neighbor_id)
                    unvisited_sites.discard(neighbor_id)
                else:
                    break
                    
                if len(cluster) >= max_bs or len(ecis) >= max_eci:
                    break
            return cluster, ecis

        while unvisited_sites:
            seed = next(iter(unvisited_sites))
            cluster = {seed}
            ecis = set(site_ecis[seed])
            cran_id_seed = site_cran.get(seed)
            visited.add(seed)
            unvisited_sites.remove(seed)

            # 统计同C-RAN邻居
            cran_members = []
            other_empty_cran = []
            other_cran = []
            
            for neighbor_id, dist in distance_dict.get(seed, []):
                if neighbor_id in visited:
                    continue
                    
                neighbor_cran = site_cran.get(neighbor_id)
                if neighbor_cran == cran_id_seed and pd.notna(cran_id_seed):
                    cran_members.append((neighbor_id, dist))
                elif pd.isna(neighbor_cran):
                    other_empty_cran.append((neighbor_id, dist))
                elif pd.notna(neighbor_cran):
                    other_cran.append((neighbor_id, dist))

            use_cran = len(cran_members) >= cran_min_members

            # Step 1: 优先加入同C-RAN邻居（仅当use_cran为True）
            if use_cran:
                cluster, ecis = add_neighbors_to_cluster(cluster, ecis, cran_members)

            # Step 2: 补充C-RAN为空的邻居
            if len(cluster) < site_num and len(ecis) < cell_num:
                cluster, ecis = add_neighbors_to_cluster(cluster, ecis, other_empty_cran)

            # Step 3: 若未满，再补充其它未访问邻居
            if len(cluster) < site_num and len(ecis) < cell_num:
                cluster, ecis = add_neighbors_to_cluster(cluster, ecis, other_cran)

            # 最终写入簇编号
            for bs_id in cluster:
                for eci in site_ecis[bs_id]:
                    eci_to_cluster[eci] = cluster_id
            cluster_id += 1

        df['簇编号'] = df['ECI'].map(eci_to_cluster)

        output_path = filepath if overwrite else os.path.join(output_folder, '簇明细-' + filename)
        try:
            df.to_csv(output_path, index=False, encoding='gbk')
            print(f"✅ 已输出：{output_path}\n")
        except Exception as e:
            print(f"❌ 写入失败：{filename}，错误信息：{e}")

def merge_cluster_results(cluster_dir, city_column='传输汇聚环'):
    """
    合并所有簇明细文件，添加唯一簇编号，并以唯一簇编号进行排名和统计。
    """
    all_rows = []
    for filename in os.listdir(cluster_dir):
        if '簇明细' in filename and filename.endswith('.csv'):
            file_path = os.path.join(cluster_dir, filename)
            try:
                df = pd.read_csv(file_path, encoding='gbk')
                all_rows.append(df)
            except Exception as e:
                print(f"❌ 无法读取文件：{filename}，错误：{e}")

    if not all_rows:
        print("⚠️ 没有找到可合并的簇明细文件。")
        return

    merged_df = pd.concat(all_rows, ignore_index=True)
    output_file = f"{cluster_dir}.csv"
    merged_df.to_csv(output_file, index=False, encoding='gbk')
    print(f"✅ 已合并所有簇明细，输出文件：{output_file}")

    # 1. 读取合并后的文件
    df = merged_df

    # 2. 删除ECI列并去重
    if 'ECI' in df.columns:
        df = df.drop(columns=['ECI'])
    df = df.drop_duplicates()

    # 3. 添加唯一簇编号
    if city_column not in df.columns:
        print(f"❌ 缺少'{city_column}'列，无法生成唯一簇编号。")
        return
    if '簇编号' not in df.columns:
        print("❌ 缺少'簇编号'列，无法生成唯一簇编号。")
        return
    df['唯一簇编号'] = df[city_column].astype(str) + '-' + df['簇编号'].astype(str)

    # 4. 统计簇内小区数量和基站数量
    cluster_stats = df.groupby('唯一簇编号').agg(
        簇内小区数量=('小区数量', 'sum'),
        簇内基站数量=('基站ID', 'nunique')
    ).reset_index()

    df = df.merge(cluster_stats, on='唯一簇编号', how='left')

    # 5. 簇内基站流量排名和簇流量排名，仅在有"日均流量"列时执行
    if '日均流量' in df.columns:
        # 5.1 簇内基站流量排名
        if '基站ID' in df.columns and '唯一簇编号' in df.columns:
            df['簇内基站流量排名'] = df.groupby('唯一簇编号')['日均流量'].rank(method='dense', ascending=False).astype(int)
        else:
            print("⚠️ 缺少'基站ID'或'唯一簇编号'列，无法进行簇内基站流量排名。")

        # 5.2 计算每个唯一簇的总流量并排名
        if '唯一簇编号' in df.columns:
            cluster_flow = df.groupby('唯一簇编号')['日均流量'].sum().reset_index()
            cluster_flow['簇流量排名'] = cluster_flow['日均流量'].rank(method='dense', ascending=False).astype(int)
            df = df.merge(cluster_flow[['唯一簇编号', '簇流量排名']], on='唯一簇编号', how='left')
        else:
            print("⚠️ 缺少'唯一簇编号'列，无法进行簇流量排名。")
    else:
        print("⚠️ 缺少'日均流量'列，跳过所有流量排名。")

    # 6. 为每个地市的唯一簇编号添加1开始的序号，并生成"地市簇编号"
    df['地市簇序号'] = df.groupby('地市')['唯一簇编号'].transform(
        lambda x: pd.factorize(x)[0] + 1)
    df['地市簇编号'] = df['地市'].astype(str) + '-' + df['地市簇序号'].astype(str)
   
    # 8. 整理数据
    # 删除过程列
    if '簇编号' in df.columns:
        df = df.drop(columns=['簇编号'])
    if '唯一簇编号' in df.columns:
        df = df.drop(columns=['唯一簇编号'])
    # 按地市簇序号升序排列
    if '地市簇序号' in df.columns:
        df = df.sort_values(by='地市簇序号', ascending=True)
    
    # 9. 保存最终结果
    final_file = f"{cluster_dir}-排名结果.xlsx"
    df.to_excel(final_file, index=False, engine='openpyxl')

    print(f"🏅 排名处理完成，输出文件：{final_file}")

def cleanup_temp_files(expanded_file, split_dir, cluster_dir):
    """
    删除中间文件和目录
    """
    try:
        if os.path.exists(expanded_file):
            os.remove(expanded_file)
            print(f'📄 已删除: {expanded_file}')
        if os.path.exists(cluster_dir):
            if os.path.isdir(cluster_dir):
                shutil.rmtree(cluster_dir)
                print(f'📂 已删除目录: {cluster_dir}')
            else:
                os.remove(cluster_dir)
                print(f'🗑️ 已删除文件: {cluster_dir}.')
        if os.path.exists(split_dir):
            if os.path.isdir(split_dir):
                shutil.rmtree(split_dir)
                print(f'📂 已删除目录: {split_dir}')
            else:
                os.remove(split_dir)
                print(f'🗑️ 已删除文件: {split_dir}.')
        print("✅ 临时文件清理完成！")
    except Exception as e:
        print(f"⚠️ 清理过程中出现错误: {e}")

def run_full_process(original_file, city_column='传输汇聚环', cran_min_members=3, 
                     neighbor_num=5000, chunk_size=300000, cell_num=24, site_num=8):
    """
    执行完整的簇划分流程
    """
    original_file_dir = os.path.dirname(original_file)
    base_name = os.path.splitext(os.path.basename(original_file))[0]

    expanded_file = os.path.join(original_file_dir, f"{base_name}-展开.csv")
    split_dir = os.path.join(original_file_dir, f"{base_name}-拆分")
    cluster_dir = os.path.join(original_file_dir, f"{base_name}-簇划分")
        
    print("🚩 步骤1：处理原始文件 ...")
    expand_by_cell_count(original_file, expanded_file)

    print("🚩 步骤2：按聚类单位拆分 ...")
    split_csv_by_city(expanded_file, split_dir, city_column=city_column, chunk_size=chunk_size)

    print("🚩 步骤3：基站信息聚类 ...")
    cluster_city_files(
        split_dir, cluster_dir,
        city_column=city_column,
        overwrite=False,
        neighbor_num=neighbor_num,
        cran_min_members=cran_min_members,
        cell_num=cell_num,
        site_num=site_num
    )

    print("🚩 步骤4：合并聚类结果 ...")
    merge_cluster_results(cluster_dir, city_column=city_column)

    # 清理临时文件
    print("\n🧹 清理临时文件...")
    cleanup_temp_files(expanded_file, split_dir, cluster_dir)

class ClusterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能簇划分工具")
        self.setMinimumSize(1000, 800)
        self.set_icon()
        self.init_ui()

        self.signals = WorkerSignals()
        self.signals.log.connect(self.append_log)
        self.signals.progress.connect(self.update_progress)
        self.signals.done.connect(self.processing_done)
        self.signals.error.connect(self.processing_error)
        self.progress_lines = {}  # filename: QTextCursor for that line

    def set_icon(self):
        # 使用系统默认图标，避免base64解码错误
        self.setWindowIcon(QApplication.style().standardIcon(QApplication.style().SP_ComputerIcon))
        
    def update_progress_line(self, filename, text):
        cursor = self.log_output.textCursor()
        doc = self.log_output.document()

        if filename in self.progress_lines:
            block = self.progress_lines[filename]
            cursor.setPosition(block.position())
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.insertText(text)
        else:
            # 插入新行
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(text + "\n")
            self.progress_lines[filename] = cursor.block()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 文件选择
        file_group = QGroupBox("文件选择")
        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        file_btn = QPushButton("浏览")
        file_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(QLabel("选择文件："))
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(file_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # 参数设置
        param_group = QGroupBox("参数设置")
        grid = QGridLayout()

        self.unit_group = QButtonGroup()
        self.radio_ds = QRadioButton("地市")
        self.radio_cr = QRadioButton("传输汇聚环")
        self.radio_cr.setChecked(True)
        self.unit_group.addButton(self.radio_ds)
        self.unit_group.addButton(self.radio_cr)

        grid.addWidget(QLabel("簇划分单位："), 0, 0)
        grid.addWidget(self.radio_ds, 0, 1)
        grid.addWidget(self.radio_cr, 0, 2)

        self.param_entries = {}
        params = [
            ("簇内最大小区数", "24", "默认值为24，建议为基站数的3倍"),
            ("簇内最大基站数", "8", "默认值为8，建议为小区数的1/3"),
            ("C-RAN阈值", "3", "C-RAN内基站数小于此阈值时忽略C-RAN聚类规则"),
            ("聚类小区数", "5000", "聚类时计算邻区聚类的最大邻区数"),
            ("每个城市文件的最大行数", "300000", "建议按地市或传输汇聚环拆分控制文件大小")
        ]

        for i, (name, default, tip) in enumerate(params):
            grid.addWidget(QLabel(name + "："), i+1, 0)
            entry = QLineEdit(default)
            self.param_entries[name] = entry
            grid.addWidget(entry, i+1, 1)
            grid.addWidget(QLabel(tip), i+1, 2)

        param_group.setLayout(grid)
        layout.addWidget(param_group)

        # 操作按钮
        op_layout = QHBoxLayout()
        run_btn = QPushButton("开始执行")
        run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        run_btn.clicked.connect(self.start_processing)
        tmpl_btn = QPushButton("生成模板（必需字段）")
        tmpl_btn.clicked.connect(self.generate_template)
        op_layout.addWidget(run_btn)
        op_layout.addWidget(tmpl_btn)
        layout.addLayout(op_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(24)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #d0d0d0;
                border-radius: 12px;
                background-color: #ffffff;
                text-align: center;
                font: bold 10pt "Microsoft YaHei";
                color: #444;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #66bb6a, stop:1 #43a047
                );
                border-radius: 10px;
                margin: 1px;
            }
        """)

        self.status_label = QLabel("状态：等待开始")
        self.status_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        # 日志输出窗口（终端风格）
        log_group = QGroupBox("运行日志")
        log_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
            }
        """)

        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color:  #ffffff;
                border: none;
                font-family: Consolas;
                font-size: 10pt;
            }
        """)
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择CSV文件", "", "CSV文件 (*.csv)")
        if path:
            self.file_input.setText(path)

    def generate_template(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存模板", "", "CSV文件 (*.csv)")
        if not path:
            return
        try:
            df = pd.DataFrame(columns=["地市", "基站ID", "小区数量", "经度", "纬度", "C-RAN标识", "传输汇聚环"])
            df.to_csv(path, index=False, encoding="gbk")
            QMessageBox.information(self, "模板生成", f"✅ 模板文件已生成：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模板生成失败：{str(e)}")

    def start_processing(self):
        path = self.file_input.text()
        if not os.path.isfile(path):
            QMessageBox.critical(self, "错误", "请选择有效的CSV文件！")
            return

        try:
            params = {
                "file": path,
                "max_cells": int(self.param_entries["簇内最大小区数"].text()),
                "max_sites": int(self.param_entries["簇内最大基站数"].text()),
                "cran_threshold": int(self.param_entries["C-RAN阈值"].text()),
                "neighbor_num": int(self.param_entries["聚类小区数"].text()),
                "chunk_size": int(self.param_entries["每个城市文件的最大行数"].text()),
                "split_unit": "地市" if self.radio_ds.isChecked() else "传输汇聚环"
            }
        except ValueError:
            QMessageBox.critical(self, "错误", "请确保所有参数都是整数！")
            return

        self.status_label.setText("状态：处理中...")
        self.progress_bar.setValue(0)
        self.log_output.clear()

        threading.Thread(target=self.run_processing, args=(params,), daemon=True).start()

    def run_processing(self, params):
        mystdout = io.StringIO()
        sys_stdout = sys.stdout
        stop_flag = threading.Event()

        def flush_logs():
            last_log = ""
            while not stop_flag.is_set():
                curr = mystdout.getvalue()
                if curr != last_log:
                    self.signals.log.emit(curr[len(last_log):])
                    last_log = curr
                time.sleep(0.2)

        try:
            sys.stdout = mystdout
            flusher = threading.Thread(target=flush_logs, daemon=True)
            flusher.start()

            run_full_process(
                original_file=params["file"],
                city_column=params["split_unit"],
                cran_min_members=params["cran_threshold"],
                neighbor_num=params["neighbor_num"],
                chunk_size=params["chunk_size"],
                cell_num=params["max_cells"],     # 添加簇内最大小区数参数
                site_num=params["max_sites"]      # 添加簇内最大基站数参数
            )

            self.signals.progress.emit(100)
            self.signals.done.emit()
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            stop_flag.set()
            sys.stdout = sys_stdout

    def append_log(self, text):
        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def update_progress(self, val):
        self.progress_bar.setValue(val)

    def processing_done(self):
        self.status_label.setText("✅ 处理完成！")

    def processing_error(self, msg):
        self.status_label.setText("❌ 执行失败")
        QMessageBox.critical(self, "执行错误", msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ClusterApp()
    win.show()

    sys.exit(app.exec_())
