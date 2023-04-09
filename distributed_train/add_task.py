import xmlrpc.client
import threading
import json
import time
import subprocess
import os
from datetime import datetime, timedelta
import paramiko
import configparser
from utils import FedNLL_name, make_exp_name, read_fednll_args, result_parser, get_command, task_has_completed
import argparse

config = configparser.ConfigParser()
config.read('./distributed_train/config.ini')
machine_name = config.get('Worker', 'machine_name')
# if machine_name == "V14":
#     server_ip .= "172.17.01"
if machine_name == "V13" or machine_name == "V14":
    server_ip = "113.54.131.71"  #  V14做server
elif machine_name == "V4" or machine_name == "V3":
    server_ip = "10.249.44.98"   # V4的ip
else:
    raise ValueError("machine_name错了")  

server = f"http://{server_ip}:8000/"

# num_threads = 0
# machine_name = "V4"

class TaskGenerator:
    def __init__(
        self, 
        dataset:str, 
        model:str, 
        criterion:str,
        sce_alpha:float,
        sce_beta:float,
        partition_all:list, 
        globalize_all:list, 
        noise_mode_all:list,
        noise_ratio_sym_all:list,
        noise_ratio_asym_all:list,
        raw_data_dir:str,
        data_dir:str,
        out_dir:str,
        com_round:int,
        epochs:int,
        seed_all:list,
        lr:float,
        momentum:float,
        weight_decay:float,
        dir_alpha:float,
        major_classes_num:int,
        num_clients=10, 
        sample_ratio=1.0,
    ):
        self.dataset = dataset
        self.model = model
        self.criterion = criterion
        self.sce_alpha = sce_alpha
        self.sce_beta = sce_beta
        self.partition_all = partition_all
        self.globalize_all = globalize_all
        self.noise_mode_all = noise_mode_all
        self.noise_ratio_sym_all = noise_ratio_sym_all
        self.noise_ratio_asym_all = noise_ratio_asym_all
        self.raw_data_dir = raw_data_dir
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.com_round = com_round
        self.epochs = epochs
        self.seed_all = seed_all
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.dir_alpha = dir_alpha
        self.major_classes_num = major_classes_num
        self.sample_ratio = sample_ratio
        self.idx = 0


    def AddTask(self, args, priority=False):
        proxy = xmlrpc.client.ServerProxy(server)
        return proxy.AddTask(args, priority)

    def DeleteTask(self, args):
        proxy = xmlrpc.client.ServerProxy(server)
        return proxy.DeleteTask(args)
        
    def generate_task(self):
        tasks = []
        for seed in self.seed_all:
            for noise_mode in self.noise_mode_all:
                for globalize in self.globalize_all:

                    if globalize:
                        if noise_mode == "clean":
                            noise_ratio_all = [0.0]
                        else:
                            noise_ratio_all = self.noise_ratio_sym_all
                    else:
                        if noise_mode == "clean":
                            continue
                        noise_ratio_all = self.noise_ratio_asym_all
                    
                    for partition in self.partition_all:
                        for noise_ratio in noise_ratio_all:
                            min_noise_ratio = 0
                            max_noise_ratio = 0
                            if not globalize:
                                min_noise_ratio = noise_ratio[0]
                                max_noise_ratio = noise_ratio[1]
                        
                            args = read_fednll_args()

                            args.dataset = self.dataset
                            args.model = self.model
                            args.globalize = globalize
                            args.partition = partition
                            args.num_clients = self.num_clients
                            args.noise_mode = noise_mode
                            args.major_classes_num = self.major_classes_num
                            args.dir_alpha = self.dir_alpha
                            args.min_noise_ratio = min_noise_ratio
                            args.max_noise_ratio = max_noise_ratio
                            args.noise_ratio = noise_ratio
                            args.lr = self.lr
                            args.momentum = self.momentum
                            args.weight_decay = self.weight_decay
                            args.com_round = self.com_round
                            args.epochs = self.epochs
                            args.seed = seed
                            args.out_dir = self.out_dir
                            args.raw_data_dir = self.raw_data_dir
                            args.data_dir = self.data_dir
                            args.sample_ratio = self.sample_ratio
                            args.criterion = self.criterion

                            nll_name = FedNLL_name(**vars(args))
                            exp_name = make_exp_name('fedavg', args)
                            if args.criterion != 'ce':
                                alg_name = 'FedAvg-RobustLoss'
                            else:
                                alg_name = 'FedAvg'
                            self.out_path = os.path.join(
                                args.out_dir, nll_name, alg_name, exp_name
                            )
                            self.record_file = os.path.join(self.out_path, 'result_record.txt')
                            # 将所有任务添加到tasks中
                            if task_has_completed(self.record_file):
                                continue
                            
                            command = get_command(args) # 根据参数构造执行指令["构造数据集的指令", "跑代码的指令"]

                            with open('./distributed_train/file.txt', 'a') as file:  
                                # 在 file 变量中写入内容  
                                file.write(f"{self.idx}:{{'command':{command},'out_path':{self.out_path}}}\n")  
                            tasks.append({'command':command, 'out_path':self.out_path})
                            self.idx+=1

        return tasks

                    
priority = False  # 如果任务需要优先完成，可以将这个值设为True
dataset = "cifar10"
model = "VGG16"
criterion = "ce"
sce_alpha = 0.01  # criterion为sce时要使用
sce_beta = 1.0    # criterion为sce时要使用
partition_all =  [ "iid", "noniid-#label", "noniid-quantity"]   # ["iid", "noniid-#label", "noniid-labeldir", "noniid-quantity"]
globalize_all = [True, False]
noise_mode_all = ["clean", "sym", "asym"]
noise_ratio_sym_all =   [0.4]  # [0.2,0.4,0.6,0.8]
noise_ratio_asym_all =  [[0.3,0.5]] # [[0.1,0.3],[0.3,0.5],[0.5,0.7]]
seed_all = [1,2,3]  # [1,2,3]
dir_alpha = 0.1    # TODO：这里是0.1
raw_data_dir = f"../raw_datasets/{dataset}"
data_dir = f"../fedNLLdata/{dataset}"
out_dir = "../Fed-Noisy-checkpoint/"
com_round = 500
epochs = 5
lr = 0.01
momentum = 0.9
major_classes_num = 3  # 这里是3
num_clients = 100
sample_ratio = 0.1  # 要改为1.0
weight_decay = 5e-4


m = TaskGenerator(
        dataset=dataset, 
        model=model,
        criterion=criterion,
        sce_alpha=sce_alpha,
        sce_beta=sce_beta,
        partition_all=partition_all,
        globalize_all=globalize_all,
        noise_mode_all=noise_mode_all,
        noise_ratio_sym_all=noise_ratio_sym_all,
        noise_ratio_asym_all=noise_ratio_asym_all,
        dir_alpha=dir_alpha,
        raw_data_dir=raw_data_dir,
        data_dir=data_dir,
        out_dir=out_dir,
        com_round=com_round,
        epochs=epochs,
        seed_all=seed_all,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        major_classes_num=major_classes_num,
        num_clients=num_clients,
        sample_ratio=sample_ratio,
)

tasks = m.generate_task()

delete = input(f'添加任务输入1,删除任务输入2,任务长度:{len(tasks)}')

if delete == '2':
    user_input = input('是否删除任务？(yes/no) ')
    if user_input == 'yes':
        reply = m.DeleteTask(tasks)
        print(f"成功删除{reply}个任务")
    elif user_input == 'no':
        pass
    else:
        print('无效的输入，程序结束。')
else:
    user_input = input('是否添加任务？(yes/no) ')
    if user_input == 'yes':
        reply = m.AddTask(tasks, priority)
        print(f"成功添加{reply}个任务")
    elif user_input == 'no':
        pass
    else:
        print('无效的输入，程序结束。')