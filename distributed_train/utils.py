import argparse
import ast
import os
import re

def FedNLL_name(
    dataset, globalize, partition, num_clients=10, noise_mode='clean', **kw
):
    if noise_mode == 'clean':
        kw['noise_ratio'] = 0.0

    prefix = 'fedNLL'
    if partition == 'noniid-#label':
        partition_param = f"{kw['major_classes_num']}"
    elif partition == 'noniid-quantity':
        partition_param = f"{kw['dir_alpha']}"
    elif partition == 'noniid-labeldir':
        partition_param = f"{kw['dir_alpha']:.2f}_{kw['min_require_size']}"
    else:
        # IID
        partition_param = ''
    partition_setting = f"{num_clients}_{partition}_{partition_param}"
    noise_setting = ''
    if globalize is False:
        noise_param = f"local_{noise_mode}_min_{kw['min_noise_ratio']:.2f}_max_{kw['max_noise_ratio']:.2f}"
    else:
        noise_param = f"global_{noise_mode}_{kw['noise_ratio']:.2f}"  # if noise_ratio is a float number
    setting = f"{partition_setting}_{noise_param}"
    return f"{prefix}_{dataset}_{setting}"

def make_exp_name(alg_name='dividemix', args=None):
    if alg_name == 'crossentropy':
        noise_name = f"noise_mode={args.noise_mode}-noise_ratio={args.noise_ratio:.2f}"
        opt_name = f"lr={args.lr:.4f}-momentum={args.momentum:.2f}-weight_decay={args.weight_decay:.5f}"
        other_name = f"num_epochs={args.num_epochs}-batch_size={args.batch_size}-seed={args.seed}"
        exp_name = '-'.join([noise_name, opt_name, other_name])

    elif alg_name == 'dividemix':
        noise_name = f"noise_mode={args.noise_mode}-noise_ratio={args.noise_ratio:.2f}"
        alg_name = f"p_threshold={args.p_threshold:.2f}-lambda_u={args.lambda_u}-T={args.T:.2f}-alpha={args.alpha:.2f}"
        opt_name = f"lr={args.lr:.4f}-momentum={args.momentum:.2f}-weight_decay={args.weight_decay:.5f}"
        other_name = f"num_epochs={args.num_epochs}-batch_size={args.batch_size}-seed={args.seed}"
        exp_name = '-'.join([noise_name, alg_name, opt_name, other_name])

    elif alg_name == 'coteaching':
        pass
    elif alg_name == 'fedavg':
        arch_name = f"arch={args.model}"
        opt_name = f"lr={args.lr:.4f}-momentum={args.momentum:.2f}-weight_decay={args.weight_decay:.5f}"
        criterion_name = make_criterion_name(args)
        other_name = f"com_round={args.com_round}-local_epochs={args.epochs}-batch_size={args.batch_size}-seed={args.seed}"
        exp_name = '-'.join([alg_name, criterion_name, arch_name, opt_name, other_name])

    return exp_name

def make_criterion_name(args):
    criterion_name = f'criterion={args.criterion}'

    if args.criterion == 'ce':
        criterion_param = ''
    elif args.criterion == 'sce':
        criterion_param = f"sce_alpha={args.sce_alpha:.2f}-sce_beta={args.sce_beta:.2f}"
    elif args.criterion in ['rce', 'nce', 'nrce']:
        criterion_param = f"loss_scale={args.loss_scale:.2f}"
    elif args.criterion == 'gce':
        criterion_param = f"gce_q={args.gce_q:.2f}"
    elif args.criterion == 'ngce':
        criterion_param = f"loss_scale={args.loss_scale:.2f}-gce_q={args.gce_q:.2f}"
    elif args.criterion in ['mae', 'nmae']:
        criterion_param = f"loss_scale={args.loss_scale:.2f}"
    elif args.criterion in ['focal', 'nfocal']:
        if args.focal_alpha is None:
            criterion_param = f"focal_gamma={args.focal_gamma:.2f}-focal_alpha=None"
        else:
            criterion_param = (
                f"focal_gamma={args.focal_gamma:.2f}-focal_alpha={args.focal_alpha:.2f}"
            )
    criterion_name = '-'.join([criterion_name, criterion_param])
    return criterion_name


def result_parser(result_path):
    """_summary_

    Args:
        result_path (str): _description_

    Returns:
        tuple[List[float], List[float], Dict]: _description_
    """
    with open(result_path, 'r') as f:
        lines = f.readlines()
    # hist accuracy
    if len(lines) <= 1:
        return [], [], {}
    accs = [float(item) for item in lines[1].strip()[5:-1].split(', ')]
    # hist losses
    losses = [float(item) for item in lines[2].strip()[6:-1].split(', ')]
    # hyperparameter setting
    setting_dict = ast.literal_eval(lines[0].strip())
    return accs, losses, setting_dict

def task_has_completed(record_file):
    if os.path.exists(record_file):
        directory = record_file.split("/")
        for file in directory:  
            if re.search(f"com_round=(\d+)", file):
                com_round = int(re.search(f"com_round=(\d+)", file).group(1))  # 从 record_file 中解析出 com_round
        accs, _, _ = result_parser(record_file)
        if len(accs) >= com_round:
            return True
    return False


def read_fednll_args():
    parser = argparse.ArgumentParser(description='Federated Noisy Labels Preparation')

    # ==== Pipeline args ====

    parser.add_argument(
        '--num_clients',
        default=10,
        type=int,
        help="Number for clients in federated setting.",
    )
    parser.add_argument("--com_round", type=int, default=3)
    parser.add_argument(
        "--model",
        type=str,
        default='ResNet18',
        help="Currently only support 'Cifar10Net', 'SimpleCNN',  'LeNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'ToyModel', 'ResNet18', 'WRN28_10', 'WRN40_2' and 'ResNet34'.",
    )
    parser.add_argument("--sample_ratio", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.0)
    # parser.add_argument("--lr_decay_per_round", type=float, default=1)

    # ==== FedNLL data args ====
    parser.add_argument(
        '--centralized',
        default=False,
        help="Centralized setting or federated setting. True for centralized "
        "setting, while False for federated setting.",
    )
    # ----Federated Partition----
    parser.add_argument(
        '--partition',
        default='iid',
        type=str,
        choices=['iid', 'noniid-#label', 'noniid-labeldir', 'noniid-quantity'],
        help="Data partition scheme for federated setting.",
    )

    parser.add_argument(
        '--dir_alpha',
        default=0.1,
        type=float,
        help="Parameter for Dirichlet distribution.",
    )
    parser.add_argument(
        '--major_classes_num',
        default=2,
        type=int,
        help="Major class number for 'noniid-#label' partition.",
    )
    parser.add_argument(
        '--min_require_size',
        default=10,
        type=int,
        help="Minimum sample size for each client.",
    )

    # ----Noise setting options----
    parser.add_argument(
        '--noise_mode',
        default=None,
        type=str,
        choices=['clean', 'sym', 'asym'],
        help="Noise type for centralized setting: 'sym' for symmetric noise; "
        "'asym' for asymmetric noise; 'real' for real-world noise. Only works "
        "if --centralized=True.",
    )
    parser.add_argument(
        '--globalize',
        action='store_true',
        help="Federated noisy label setting, globalized noise or localized noise.",
    )

    parser.add_argument(
        '--noise_ratio',
        default=0.0,
        type=float,
        help="Noise ratio for symmetric noise or asymmetric noise.",
    )
    parser.add_argument(
        '--min_noise_ratio',
        default=0.0,
        type=float,
        help="Minimum noise ratio for symmetric noise or asymmetric noise. Only works when 'globalize' is Flase",
    )
    parser.add_argument(
        '--max_noise_ratio',
        default=1.0,
        type=float,
        help="Maximum noise ratio for symmetric noise or asymmetric noise. Only works when 'globalize' is Flase",
    )

    # ----Robust Loss Function options----
    parser.add_argument(
        "--criterion", type=str, default='ce'
    )  # for robust loss function
    parser.add_argument(
        "--sce_alpha",
        type=float,
        default=0.1,
        help="Symmetric cross entropy loss: alpha * CE + beta * RCE",
    )
    parser.add_argument(
        "--sce_beta",
        type=float,
        default=1.0,
        help="Symmetric cross entropy loss: alpha * CE + beta * RCE",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=1.0,
        help="scale parameter for loss, for example, scale * RCE, scale * NCE, scale * normalizer * RCE.",
    )
    parser.add_argument(
        "--gce_q",
        type=float,
        default=0.7,
        help="q parametor for Generalized-Cross-Entropy, Normalized-Generalized-Cross-Entropy.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=None,
        help="alpha parameter for Focal loss and Normalzied Focal loss.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=0.0,
        help="gamma parameter for Focal loss and Normalzied Focal loss.",
    )

    # ----Path options----
    parser.add_argument(
        '--dataset',
        default='cifar10',
        type=str,
        choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'clothing1m', 'webvision'],
        help="Dataset for experiment. Current support: ['mnist', 'cifar10', "
        "'cifar100', 'svhn', 'clothing1m', 'webvision']",
    )
    # parser.add_argument(
    #     '--raw_data_dir',
    #     default='../data',
    #     type=str,
    #     help="Directory for raw dataset download",
    # )
    parser.add_argument(
        '--data_dir',
        default='../noisy_label_data',
        type=str,
        help="Directory to save the dataset with noisy labels.",
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='../checkponit/',
        help="Checkpoint path for log files and report files.",
    )

    # ----Miscs options----
    parser.add_argument(
        "--save_best", action='store_true', help="Whether to save the best model."
    )
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    return args


def get_command(args):
    command = []
    data_generate_command = "python"
    data_generate_command += " build_dataset_fed.py"
    data_generate_command += " --dataset " + args.dataset
    data_generate_command += " --partition " + args.partition
    data_generate_command += " --num_clients " + str(args.num_clients)
    if args.partition == "noniid-labeldir" or args.partition == "noniid-quantity":
        data_generate_command += " --dir_alpha " + str(args.dir_alpha)
    if args.globalize:
        data_generate_command += " --globalize"
    data_generate_command += " --noise_mode " + args.noise_mode
    data_generate_command += " --raw_data_dir " + args.raw_data_dir
    data_generate_command += " --data_dir " + args.data_dir
    data_generate_command += " --seed " + str(args.seed)
    if args.globalize:
        data_generate_command += " --noise_ratio " + str(args.noise_ratio)
    else:
        data_generate_command += " --min_noise_ratio " + str(args.min_noise_ratio)
        data_generate_command += " --max_noise_ratio " + str(args.max_noise_ratio)
    if args.partition == "noniid-#label":
        data_generate_command += " --major_classes_num " + str(args.major_classes_num)
    command.append(data_generate_command)


    standalone_command = "python"
    standalone_command += " fednoisy/algorithms/fedavg/main.py"
    standalone_command += " --dataset " + args.dataset
    standalone_command += " --model " + args.model
    standalone_command += " --partition " + args.partition
    standalone_command += " --num_clients " + str(args.num_clients)
    if args.partition == "noniid-labeldir" or args.partition == "noniid-quantity":
        standalone_command += " --dir_alpha " + str(args.dir_alpha)
    if args.globalize:
        standalone_command += " --globalize"
    standalone_command += " --noise_mode " + args.noise_mode
    if args.globalize:
        standalone_command += " --noise_ratio " + str(args.noise_ratio)
    else:
        standalone_command += " --min_noise_ratio " + str(args.min_noise_ratio)
        standalone_command += " --max_noise_ratio " + str(args.max_noise_ratio)
    standalone_command += " --data_dir " + args.data_dir
    standalone_command += " --out_dir " + args.out_dir
    standalone_command += " --com_round " + str(args.com_round)
    standalone_command += " --epochs " + str(args.epochs)
    standalone_command += " --sample_ratio " + str(args.sample_ratio)
    standalone_command += " --lr " + str(args.lr)
    standalone_command += " --momentum " + str(args.momentum)
    standalone_command += " --weight_decay " + str(args.weight_decay)
    standalone_command += " --seed " + str(args.seed)
    if args.partition == "noniid-#label":
        standalone_command += " --major_classes_num " + str(args.major_classes_num)

    standalone_command += " --criterion " + args.criterion
    if args.criterion == "sce":
        standalone_command += " --sce_alpha " + str(args.sce_alpha)
        standalone_command += " --sce_beta " + str(args.sce_beta)

    command.append(standalone_command)
    return command


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