import threading
import json
import enum
import os
from datetime import datetime, timedelta
import time
import queue
import configparser
from collections import deque

from xmlrpc.server import SimpleXMLRPCServer

from utils import FedNLL_name, make_exp_name, read_fednll_args, result_parser, get_command, task_has_completed

import logging


config = configparser.ConfigParser()
config.read('./distributed_train/config.ini')
machine_name = config.get('Worker', 'machine_name')
# if machine_name == "V14":
#     server_ip .= "172.17.01"
if machine_name == "V13" or machine_name == "V14":
    remote_path = '/home/liangsiqi/Fed-Noisy-checkpoint'  #  V14做server
elif machine_name == "V4" or machine_name == "V3":
    remote_path = '/data/jintao/Fed-Noisy-checkpoint'   # V4的ip
else:
    raise ValueError("machine_name错了")  

# 配置日志输出到文件
logging.basicConfig(filename='./distributed_train/example.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

class TaskStatus(enum.Enum):
    TaskIdle = 1
    TaskInProgress = 2
    TaskCompleted = 3

class Coordinator:
    def __init__(self):

        self.tasks = {}
        # 创建一个空的队列
        self.d = deque()
        self.lock = threading.Lock()
        self.done = False
        self.idx = 0  # 任务编号
        self.task_cnt = 0  # 记录总任务数
        self.done_cnt = 0
        self.tasks_set = set()
        self.begin_time = datetime.now()
   

    def periodicTask(self):
        # 循环检测是否有任务因为卡住没做完，将卡死的任务重新加回空闲队列
        while True:
            flag = True
            for task_id, task_data in self.tasks.items():
                if task_data["status"] != TaskStatus.TaskCompleted:
                    flag = False
                if task_data["status"] == TaskStatus.TaskInProgress and (datetime.now() - task_data["last_log_time"] > timedelta(minutes=12)):
                    logging.warning(f"在periodicTask中{task_id}任务重新被加入到空闲队列中")
                    self.tasks[task_id]["status"] = TaskStatus.TaskIdle
            self.done = flag
            time.sleep(10)

    def initialize(self):
        periodicTask_Thread = threading.Thread(target=self.periodicTask)
        periodicTask_Thread.start()

    def get_idle_task(self):
        if not self.d:   # d 空
            return -1
        else:
            while True:
                if not self.d:
                    return -1
                with self.lock:
                    task_id = self.d.popleft()   # 这里取完之后得判断一下是否任务已经完成
                if self.tasks[task_id]['status'] != TaskStatus.TaskIdle:
                    continue   # 被DeleteTask设为了已完成，直接跳过
                record_file = os.path.join(self.tasks[task_id]['out_path'], 'result_record.txt')
                if task_has_completed(record_file):  # 已经有result文件了，且轮次训练满了
                    continue
                return task_id

    def AssignTask(self, args:dict):
        reply = {}
        task_id = self.get_idle_task() # 遍历tasks，找到第一个self.tasks[task_id]["status"] == idle的任务，返回task
        if task_id == -1:
            return {} # 通知client，暂时没有任务可以分配
        reply['command'] = self.tasks[task_id]['command']   
        reply['out_path'] = self.tasks[task_id]['out_path']
        reply['task_id'] = task_id
        reply['remote_path'] = remote_path
        self.tasks[task_id]["status"] = TaskStatus.TaskInProgress
        self.tasks[task_id]["last_log_time"] = datetime.now()
        
        return reply

    def get_estimated_finish_time(self):
        begin_time = self.begin_time
        # 计算已完成任务所花费的总时间
        elapsed_time = datetime.now() - begin_time

        # 计算已完成任务的平均时间
        avg_time_per_task = elapsed_time / self.done_cnt

        # 计算还需完成任务的数量
        remaining_cnt = self.task_cnt - self.done_cnt

        # 计算预估完成时间
        estimated_finish_time = datetime.now() + remaining_cnt * avg_time_per_task

        return estimated_finish_time.strftime("%Y-%m-%d %H:%M:%S")

    def TaskComplete(self, args):
        if args['success']:
            self.done_cnt+=1
            logging.info(f"[{args['machine']}]在gpu[{args['gpu_id']}]上完成[{args['task_id']}]号任务,进度[{self.done_cnt}/{self.task_cnt}],预计完成时间:{self.get_estimated_finish_time()}")
            self.tasks[args['task_id']]["status"] = TaskStatus.TaskCompleted

            # 检测是否所有任务都完成了
            if self.done_cnt == self.task_cnt:
                self.done = True
        else:
            logging.warning(f"[{args['machine']}]在gpu[{args['gpu_id']}]上任务[{args['task_id']}]失败, error:{args['error']},运行指令:{self.tasks[args['task_id']]['command'][1]}")
            self.tasks[args['task_id']]["status"] = TaskStatus.TaskIdle
            with self.lock:
                self.d.append(args['task_id'])
        return True

    def Heartbeat(self, args:dict):
        # 收到client的心跳，更新
        if 'last_log_time' not in args:
            return False
        if self.tasks[args['task_id']]['status'] == TaskStatus.TaskCompleted:  # 被DeleteTask设置为了TaskCompleted,返回True通知worker kill掉该进程
            return True

        self.tasks[args['task_id']]['last_log_time'] = datetime.strptime(args['last_log_time'], "%Y-%m-%d%H:%M:%S") # args['last_log_time'] 是 “2023-03-23 13:22:52”
        print(f"收到{args['task_id']}心跳，{self.tasks[args['task_id']]['last_log_time']}")
        # 如果发过来的上一次日志距离现在超过10分钟了，则认为该任务已挂掉，返回kill = True，提醒client杀掉该进程，同时将该任务状态设为idle
        last_log_time = self.tasks[args['task_id']]["last_log_time"]
        if datetime.now() - last_log_time > timedelta(minutes=10) and self.tasks[args['task_id']]["status"] == TaskStatus.TaskInProgress:
            logging.warning(f"在heartbeat中{args['task_id']}任务重新被加入到空闲队列中")
            self.tasks[args['task_id']]["status"] = TaskStatus.TaskIdle
            with self.lock:
                self.d.append(args['task_id'])
            return True
        return False
        
    def AddTask(self, args:dict, priority):
        old_idx = self.idx
        for item in args:
            if item['command'][1] not in self.tasks_set:    # 避免重复添加相同的任务
                self.tasks_set.add(item['command'][1])
                self.tasks[self.idx] = {'command':item['command'],'status': TaskStatus.TaskIdle, 'last_log_time': datetime.now(), 'out_path':item['out_path']}
                with self.lock:
                    if priority:
                        self.d.appendleft(self.idx)
                    else:
                        self.d.append(self.idx)
                self.idx += 1
                print(self.idx)
        if self.idx > old_idx:
            self.done = False
            logging.info(f"新增了{self.idx - old_idx}个任务")
        self.task_cnt += (self.idx - old_idx)
        return self.idx - old_idx

    def DeleteTask(self, args:dict):
        delete_cnt = 0
        for item in args:
            for task_id, task_data in self.tasks.items():
                if task_data['command'][1] == item['command'][1] and self.tasks[task_id]['status'] != TaskStatus.TaskCompleted:
                    print("删除", delete_cnt)
                    self.tasks_set.remove(task_data['command'][1])   # 把该任务从self.tasks_set中移出来
                    self.tasks[task_id]['status'] = TaskStatus.TaskCompleted
                    delete_cnt += 1
        logging.info(f"删除了{delete_cnt}个任务")
        self.task_cnt -= delete_cnt
        return delete_cnt
    
    def LogTask(self, args:dict):
        logging.info(f"将任务{args['task_id']}分配给了[{args['machine']}]上的[{args['gpu_id']}]号GPU,gpu_uuid:{args['gpu_uuid']},pid:{args['pid']}, 运行指令:{self.tasks[args['task_id']]['command'][1]}")
        return {}

    def server(self):
        self.server_handler = SimpleXMLRPCServer(("0.0.0.0", 8000))
        self.server_handler.register_function(self.AssignTask, "AssignTask")
        self.server_handler.register_function(self.TaskComplete, "TaskComplete")
        self.server_handler.register_function(self.Heartbeat, "Heartbeat")
        self.server_handler.register_function(self.AddTask, "AddTask")
        self.server_handler.register_function(self.DeleteTask, "DeleteTask")
        self.server_handler.register_function(self.LogTask, "LogTask")

        self.server_handler.serve_forever()

        # server_thread = threading.Thread(target=self.server_handler.serve_forever)
        # server_thread.daemon = True
        # server_thread.start()

    def Done(self):
        return self.done


def MakeCoordinator():
    
    m = Coordinator()
    m.initialize()
    m.server()
    return m


def main():

    m = MakeCoordinator()
    # while True:
    #     if m.Done() == True:
    #         print("目前没有任务")
    #     else:
    #         print(f"目前未完成任务数:{m.task_cnt - m.done_cnt},正在做的任务数:{m.task_cnt}")
    #     time.sleep(10)

main()