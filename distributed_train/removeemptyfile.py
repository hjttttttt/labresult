import os

def remove_file(file_path):
    print(file_path)
    try:
        os.remove(file_path)
        print(f"文件已成功删除：{file_path}")
    except OSError as e:
        print(f"文件删除失败：{file_path}，错误信息：{e}")

def remove_empty_folder(folder_path):
    try:
        os.rmdir(folder_path)
        print(f"文件夹已成功删除：{folder_path}")
        path_list = folder_path.split("/")
        previous_path = "/".join(path_list[:-1])
        print(previous_path)
        remove_empty_folder(previous_path)
    except OSError as e:
        print(f"文件夹删除失败：{folder_path}，错误信息：{e}")
        # 文件夹不为空，进一步判断
        file_list = os.listdir(folder_path)
        if len(file_list) == 0:
            # 文件夹为空，删除文件夹
            os.rmdir(folder_path)
            print(f"文件夹已成功删除：{folder_path}")
            path_list = folder_path.split("/")
            previous_path = "/".join(path_list[:-1])
            print(previous_path)
            remove_empty_folder(previous_path)
        else:
            print(f"文件夹不为空：{folder_path}")

out_path = "../Fed-Noisy-checkpointcopy/fedNLL_svhn_10_iid__global_sym_0.60/FedAvg/fedavg-criterion=ce--arch=VGG16-lr=0.0100-momentum=0.90-weight_decay=0.00050-com_round=500-local_epochs=5-batch_size=128-seed=1"
folder_path = "../Fed-Noisy-checkpointcopy/fedNLL_svhn_10_iid__global_sym_0.60/FedAvg/fedavg-criterion=ce--arch=VGG16-lr=0.0100-momentum=0.90-weight_decay=0.00050-com_round=500-local_epochs=5-batch_size=128-seed=1"
for file in os.listdir(out_path):
    remove_file(os.path.join(out_path, file))
remove_empty_folder(folder_path)