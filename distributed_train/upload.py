import xmlrpc.client
import threading
import json
import time
import subprocess
import os
import paramiko
import configparser

username = "liangsiqi"
password = "liangsiqi123"
server_ip = "113.54.131.71"  #  V14做server
# server_ip = "1.1.1.1"  #  V14做server

def upLoadLog(out_path, remote_path):
    # 远程服务器信息
    hostname = server_ip
    # 建立SSH连接
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("1")
    ssh.connect(hostname, username=username, password=password)
    # ssh.load_system_host_keys()
    # ssh.connect(hostname, username=username, key=password)
    print("2")
    # 本地文件路径和远程文件路径
    # remote_path = '/data/jintao/Fed-Noisy-checkpoint'

    absolute_path = os.path.abspath(os.path.join(remote_path, out_path))

    print(absolute_path)
    ssh.exec_command(f'mkdir -p {absolute_path}')

    for file in os.listdir(out_path):
        # 传输文件
        sftp = ssh.open_sftp()

        # try:
        #     sftp.stat(absolute_path)
        # except FileNotFoundError:
        #     sftp.mkdir(absolute_path)

        print("?")
        sftp.put(os.path.join(out_path, file), os.path.join(absolute_path, file))
        print(os.path.join(out_path, file))
        print("!!!",os.path.join(absolute_path, file))
        sftp.close()



    # 关闭SSH连接
    ssh.close()


upLoadLog('../Fed-Noisy-checkpoint/fedNLL_svhn_10_noniid-quantity_0.1_global_clean_0.00/FedAvg/fedavg-criterion=ce--arch=VGG16-lr=0.0100-momentum=0.90-weight_decay=0.00050-com_round=500-local_epochs=5-batch_size=128-seed=1','/home/liangsiqi/Fed-Noisy-checkpoint')