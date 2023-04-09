import xmlrpc.client
import threading
import json
import time
import subprocess
import os
import paramiko
import socket
import configparser
from datetime import datetime, timedelta

# def upLoadLog():
#     # 远程服务器信息
#     hostname = '10.249.44.98'
#     username = 'jintao'
#     password = '764695'

#     # 建立SSH连接
#     ssh = paramiko.SSHClient()
#     ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     ssh.connect(hostname, username=username, password=password)



#     sftp = ssh.open_sftp()


#     sftp.put('test.txt', '/data/jintao/318code/distributed_train/test2s.txt')


#     sftp.close()

#     # 关闭SSH连接
#     ssh.close()


# ip_address = socket.gethostbyname("enp5s0")

# print(ip_address)

# 创建一个配置解析器对象
# while True:
#     config = configparser.ConfigParser()

#     # 读取配置文件
#     config.read('./distributed_train/config.ini')

#     available_gpu = config.get('Worker', 'available_gpu')

#     available_gpu = eval(available_gpu)

#     print(available_gpu)
#     time.sleep(5)

# from datetime import datetime, timedelta

# # 记录程序开始时间
# begin_time = datetime.now() - timedelta(hours=3)

# # 假设已完成done_cnt个任务，总共需要完成idx个任务
# done_cnt = 100
# idx = 1000

# # 计算已完成任务所花费的总时间
# elapsed_time = datetime.now() - begin_time

# # 计算已完成任务的平均时间
# avg_time_per_task = elapsed_time / done_cnt

# # 计算还需完成任务的数量
# remaining_cnt = idx - done_cnt

# # 计算预估完成时间
# estimated_finish_time = datetime.now() + remaining_cnt * avg_time_per_task

# # 打印预估完成时间
# print("预估完成时间：", estimated_finish_time)

# import logging

# # 配置日志输出到文件
# logging.basicConfig(filename='./distributed_train/example.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

# # 输出日志
# logging.debug('This is a debug message')
# logging.info(f'This is an info message,预估完成时间: {estimated_finish_time.strftime("%Y-%m-%d %H:%M:%S")}')
# logging.warning('This is a warning message')
# logging.error('This is an error message')
# logging.critical('This is a critical message')


# import os
# import shutil

# # 设置两个目录的路径
# src_dir = 'dir_b'
# dst_dir = 'dir_a'

# print("开始复制")

# # 遍历源目录下的所有log和txt文件
# for root, dirs, files in os.walk(src_dir):
#     for file in files:
#         print(file)
#         if file.endswith('.log') or file.endswith('.txt'):
#             src_path = os.path.join(root, file)
#             dst_path = src_path.replace(src_dir, dst_dir)
#             if not os.path.exists(dst_path):
#                 print(f'Copying {src_path} to {dst_path}')
#                 os.makedirs(os.path.dirname(dst_path), exist_ok=True)
#                 shutil.copy2(src_path, dst_path)

# print("结束复制")


# import xmlrpc.client
# import threading
# import json
# import time
# import subprocess
# import os
# import paramiko
# import configparser

# server = "http://10.249.44.98:8000/"

# args = {}
# args['last_log_time'] = '2023-04-0101:01:54,488'

# args['task_id'] = 1

# # args = {'last_log_time': '2023-04-0101:15:51,844', 'task_id': 1}
    

# proxy = xmlrpc.client.ServerProxy(server)
# kill = proxy.Heartbeat(args)

import re

def task_has_completed(record_file):
    if os.path.exists(record_file):
        directory = record_file.split("/")
        print("!!!!!!!!!!!!!!!!!!!!", directory)
        for file in directory:  
            print(file)
            if re.search(f"com_round=(\d+)", file):
                com_round = int(re.search(f"com_round=(\d+)", file).group(1))  # 从 record_file 中解析出 com_round
    print("com_round:", com_round)

task_has_completed('/home/liangsiqi/Fed-Noisy-checkpoint/fedNLL_cifar10_10_iid__local_sym_min_0.30_max_0.50/FedAvg/fedavg-criterion=ce--arch=VGG13-lr=0.0100-momentum=0.90-weight_decay=0.00050-com_round=3-local_epochs=5-batch_size=128-seed=1/result_record.txt')