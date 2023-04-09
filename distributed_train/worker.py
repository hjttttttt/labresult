import xmlrpc.client
import threading
import json
import time
import subprocess
import os
import paramiko
import configparser

from utils import remove_file, remove_empty_folder

# def worker():

num_threads = 0
config = configparser.ConfigParser()
config.read('./distributed_train/config.ini')
machine_name = config.get('Worker', 'machine_name')
# if machine_name == "V14":
#     server_ip .= "172.17.01"
if machine_name == "V13" or machine_name == "V14":
    username_ = "liangsiqi"
    password_ = "liangsiqi123"
    server_ip = "113.54.131.71"  #  V14做server
elif machine_name == "V4" or machine_name == "V3":
    username_ = "jintao"
    password_ = "764695"
    server_ip = "10.249.44.98"   # V4的ip
else:
    raise ValueError("machine_name错了")  

server = f"http://{server_ip}:8000/"
    
def upLoadLog(out_path, remote_path):
    # 远程服务器信息
    hostname = server_ip
    username = username_
    password = password_

    # 建立SSH连接
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=username, password=password)

    absolute_path = os.path.abspath(os.path.join(remote_path, out_path))

    ssh.exec_command(f'mkdir -p {absolute_path}')

    for file in os.listdir(out_path):
        # 传输文件
        sftp = ssh.open_sftp()

        sftp.put(os.path.join(out_path, file), os.path.join(absolute_path, file))

        # remove_file(os.path.join(out_path, file))

        sftp.close()

    # remove_empty_folder(out_path)
    # 关闭SSH连接
    ssh.close()



def AssignTask(args: dict):
    proxy = xmlrpc.client.ServerProxy(server)
    reply = proxy.AssignTask(args)
    return reply

def LogTask(args: dict):
    proxy = xmlrpc.client.ServerProxy(server)
    reply = proxy.LogTask(args)
    return reply

def TaskComplete(args):
    proxy = xmlrpc.client.ServerProxy(server)
    proxy.TaskComplete(args)
    

def send_heartbeat(reply, stop_flag, proc, task_id):
    while not stop_flag.is_set():
        args = {}
        out_path = reply['out_path']
        record_file = os.path.join(out_path, 'server.log')

        try:
            with open(record_file, 'r') as file:
                lines = file.readlines()
                if not lines:
                    time.sleep(5)
                    continue
                last_line = lines[-1].strip()
        except:
            time.sleep(5)
            continue
        
        timestamp_str = last_line.split(' - ')[0]
        args['last_log_time'] = timestamp_str.replace(' ', '').split(',')[0]
        args['task_id'] = task_id

        proxy = xmlrpc.client.ServerProxy(server)
        kill = proxy.Heartbeat(args)

        if kill:
            proc.kill()  # 结束进程

        time.sleep(10)


def worker(gpu_uuid, gpu_id):
    while True:
        # 这里要打开某个文件，查看自己的gpu_id是否还在该文件中，不在的话就退出，可以动态退出
        config = configparser.ConfigParser()

        # 读取配置文件
        config.read('./distributed_train/config.ini')

        available_gpu = eval(config.get('Worker', 'available_gpu'))

        if gpu_uuid in available_gpu.keys():
            print(f"GPU_uuid: {gpu_uuid}, gpu:{gpu_id} is available")
        else:
            print(f"GPU_uuid: {gpu_uuid}, gpu:{gpu_id} is not available anymore")
            global num_threads
            num_threads -= 1
            exit(0)


        args = {"machine":machine_name, "gpu_id": gpu_id}
        reply = AssignTask(args)
        # 得到reply字典，判断是否分配到任务，如果分配到任务，开启一个心跳进程不断发送心跳，在任务结束后调用rpc请求通知服务器自己完成任务，
        # 否则睡眠1分钟，再次向服务器申请分配任务
        stop = False
        if 'command' in reply:
            command = reply['command']

            result = subprocess.run(command[0].split())   # 数据划分指令

            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))

            proc = subprocess.Popen(command[1].split(), env=env, stderr=subprocess.PIPE)   # 训练指令

            args_info = {"machine":machine_name, "gpu_id": gpu_id, "pid": proc.pid, 'task_id': reply['task_id'], "gpu_uuid":gpu_uuid}

            LogTask(args_info)

            time.sleep(30)  # 睡20s后再开始检测

            # 开启一个心跳线程
            stop_flag = threading.Event()
            heartbeat_thread = threading.Thread(target=send_heartbeat, args=(reply, stop_flag, proc, reply['task_id']))
            heartbeat_thread.start()

            # 等待进程完成
            output, error = proc.communicate()

            if proc.returncode == 0:
                # 将三个日志文件传输过去
                
                if machine_name == "V3" or machine_name == "V13":  # V14和V3机器需要传到服务器
                    upLoadLog(reply['out_path'], reply['remote_path']) 
                args = {'task_id': reply['task_id'], "success": True, "machine": machine_name, "gpu_id": gpu_id}
                TaskComplete(args)
            else :
                args = {'task_id': reply['task_id'], "success": False, "machine": machine_name, "gpu_id": gpu_id, "error": error.decode('utf-8')}
                TaskComplete(args)
            stop_flag.set()
        else:
            time.sleep(60)



def main():
    config = configparser.ConfigParser()


    available_gpu = []
    num_threads = len(available_gpu)
    while True:
        # 读取配置文件
        config.read('./distributed_train/config.ini')
        new_available_gpu = eval(config.get('Worker', 'available_gpu'))
        added_gpu = set(new_available_gpu) - set(available_gpu)
        if added_gpu:
            for gpu_uuid in added_gpu:
                print(f"Detected new available GPU(s): {new_available_gpu[gpu_uuid]}, gpu_uuid:{gpu_uuid}")
                t = threading.Thread(target=worker, args=(gpu_uuid, new_available_gpu[gpu_uuid],))
                t.start()
                num_threads += 1

        available_gpu = new_available_gpu

        # Check if all threads have finished
        if num_threads == 0:
            sys.exit()
        
        # Wait for a bit before checking again
        time.sleep(10)

main()