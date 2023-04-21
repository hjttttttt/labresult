import os
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import os


path = "/data/jintao/Fed-Noisy-checkpoint"

file_list = os.listdir(path)
pattern = r"acc:\s*\[([\d.,\s]+)\]"

a = 0

settings_name = []
acc_lists = []

for file_name in file_list:
   if file_name[7:12] == 'cifar':
      full_file_path = os.path.join(path, file_name)
      file_list = os.listdir(full_file_path)
      if len(file_list) != 1:
         continue
      full_file_path = os.path.join(full_file_path, file_list[0])
      file_list = os.listdir(full_file_path)
      for results_name in file_list:
        if results_name == "result_record.txt":
            full_file_path = os.path.join(full_file_path, results_name)
            with open(full_file_path, "r") as f:
               contents = f.read()
               result = re.search(pattern, contents)
               acc_str = result.group(1)
               acc_list = [float(x) for x in acc_str.split(",")]
               if len(acc_list) != 1000 :
                  continue
               settings_name.append(file_name)
               acc_lists.append(acc_list)
               a += 1


print(len(settings_name))
print(len(acc_lists))
assert(len(settings_name)==len(acc_lists))
# plt.figure(figsize=(9, 6))
# for i in range(len(settings_name)):
#     plt.plot(acc_lists[i], label=settings_name[i])

# plt.title("Mnist_acc")
# plt.xlabel("epoch")
# plt.ylabel("acc")
# plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=4)
# plt.savefig("Mnist_acc.png")


with open("Cifar_acc.txt", "w") as f:
    for i, setting in enumerate(settings_name):
        max_acc = max(acc_lists[i])
        last_acc = acc_lists[i][-1]
        f.write("{}，最高准确率：{:.2f}%，最后一轮准确率：{:.2f}%\n".format(setting, max_acc, last_acc))