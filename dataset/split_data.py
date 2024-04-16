import os
import random
import shutil
import csv

import os
import random
import shutil
import csv

def split_files(input_folder, input_csv, output_folder1, output_folder2, split_ratio=0.5):
    # 读取 CSV 文件并忽略首行
    with open(input_csv, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # 获取首行
        rows = list(csv_reader)
        
    # 获取文件名列表和标签列表
    file_names = [row[0] for row in rows]
    
    # 打乱索引列表
    indexes = list(range(len(file_names)))
    random.shuffle(indexes)
    
    # 确定分割点
    split_point = int(len(indexes) * split_ratio)
    
    # 分割索引列表
    indexes_part1 = indexes[:split_point]
    indexes_part2 = indexes[split_point:]
    
    # 将文件复制到输出文件夹1
    for idx in indexes_part1:
        file = file_names[idx]
        source_path = os.path.join(input_folder, file)
        destination_path = os.path.join(output_folder1, file)
        shutil.copyfile(source_path, destination_path)
        
    # 将文件复制到输出文件夹2
    for idx in indexes_part2:
        file = file_names[idx]
        source_path = os.path.join(input_folder, file)
        destination_path = os.path.join(output_folder2, file)
        shutil.copyfile(source_path, destination_path)
    
    # 分割 CSV 数据
    rows_part1 = [rows[idx] for idx in indexes_part1]
    rows_part2 = [rows[idx] for idx in indexes_part2]
    
    # 写入分割后的 CSV 文件并添加首行
    with open(os.path.join('label_train.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)  # 添加首行
        csv_writer.writerows(rows_part1)
        
    with open(os.path.join('label_test.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)  # 添加首行
        csv_writer.writerows(rows_part2)

# 示例用法
input_folder = "image_original"
input_csv = "label_original.csv"
output_folder1 = "image_train"
output_folder2 = "image_test"

# 创建输出文件夹
os.makedirs(output_folder1, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)

# 分割文件夹中的文件和 CSV 文件
split_files(input_folder, input_csv, output_folder1, output_folder2, split_ratio=0.8)
