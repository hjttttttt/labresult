import os
import re
import matplotlib.pyplot as plt

# 用于解析文件名中的参数
plt.figure(figsize=(20,10))
def parse_filename(filename):
    pattern = r"arch=(\w+).*lr=([\d\.]+).*weight_decay=([\d\.]+)"
    match = re.search(pattern, filename)
    if match:
        arch, lr, weight_decay = match.groups()
        return arch, float(lr), float(weight_decay)
    else:
        print("!")
        return None

# 遍历文件夹，获取所有的result_record.txt文件
result_files = []
for folder in os.listdir('.'):
    if folder.startswith('fedavg-criterion=ce--arch=') and os.path.isdir(folder):

        result_file = os.path.join(folder, 'result_record.txt')
        if os.path.exists(result_file):
            result_files.append(result_file)

i = 0
# 解析文件名中的参数，并绘制对应的曲线
for result_file in result_files:
    i += 1
    if i <= 2:
        continue
    with open(result_file, 'r') as f:
        content = f.read()
        acc_pattern = re.compile('acc:\[(.*?)\]')
        acc_match = re.search(acc_pattern, content)
        if acc_match:
    # 将结果字符串转换为列表
            acc_list = [float(x) for x in acc_match.group(1).split(',')]
    arch, lr, weight_decay = parse_filename(result_file)
    max_acc = 0
    for i in acc_list:
        max_acc = max(max_acc, i)
    label = f'{arch}, lr={lr}, weight_decay={weight_decay}, max_acc={max_acc}, last_acc={acc_list[len(acc_list)-1]}'
    plt.plot(acc_list, label=label)

# 添加图例，坐标轴标签等
plt.legend()
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Round')
plt.savefig("1.jpg")