import os
import sys
sys.path.append(os.getcwd())
import argparse
from collections import OrderedDict
import glog as log
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.interpolate import make_interp_spline
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

from config import MODELS_TEST_STANDARD
import re
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif']=['simsun'] #显示中文标签
rcParams['axes.unicode_minus']=False   #这两行需要手动设置

rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

linestyle_dict = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ("dashdot","dashdot"),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])


def read_json_data(json_path):
    # data_key can be query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query
    print("begin read {}".format(json_path))
    with open(json_path, "r") as file_obj:
        data_txt = file_obj.read()
        data_json = json.loads(data_txt)
        distortion_dict = data_json["distortion"]
        correct_all = np.array(data_json["correct_all"]).astype(np.bool)
        #success_all = np.array(data_json["success_all"]).astype(np.int32)
    return distortion_dict,  correct_all

def read_all_data(dataset_path_dict, arch, query_budgets, stats="mean_distortion"):
    # dataset_path_dict {("CIFAR-10","l2","untargeted"): "/.../"， }
    data_info = {}
    extract_pattern = "{}_(.*?)_result.json".format(arch)
    extract_pattern = re.compile(extract_pattern)
    for (dataset, norm, targeted, method), dir_path in dataset_path_dict.items():
        for file_path in os.listdir(dir_path):
            if arch in file_path and file_path.endswith(".json") and not file_path.startswith("tmp"):
                ma = re.match(extract_pattern, file_path)
                assert ma is not None, "ma in {} is None".format(dir_path + "/" + file_path)
                initial_schedule = ma.group(1)
                file_path = dir_path + "/" + file_path
                distortion_dict, correct_all = read_json_data(file_path)
                x = []
                y = []
                for query_budget in query_budgets:
                    distortion_list = []
                    for image_id, query_distortion_dict in distortion_dict.items():
                        query_distortion_dict = {int(float(query)): float(dist) for query, dist in query_distortion_dict.items()}
                        queries = np.array(list(query_distortion_dict.keys()))
                        queries = np.sort(queries)
                        find_index = np.searchsorted(queries, query_budget, side='right') - 1
                        if query_budget < queries[find_index]:
                            print(
                                "query budget is {}, find query is {}, min query is {}, len query_distortion is {}".format(
                                    query_budget, queries[find_index], np.min(queries).item(),
                                    len(query_distortion_dict)))
                            continue
                        distortion_list.append(query_distortion_dict[queries[find_index]])
                    distortion_list = np.array(distortion_list)
                    distortion_list = distortion_list[~np.isnan(distortion_list)]  # 去掉nan的值
                    mean_distortion = np.mean(distortion_list)
                    median_distortion = np.median(distortion_list)
                    x.append(query_budget)
                    if stats == "mean_distortion":
                        y.append(mean_distortion)
                    elif stats == "median_distortion":
                        y.append(median_distortion)
                x = np.array(x)
                y = np.array(y)
                data_info[(dataset, norm, targeted, initial_schedule)] = (x,y)

    return data_info




method_name_to_paper = {"tangent_attack":"Tangent Attack"} # "HSJA":"HopSkipJumpAttack",
                       # "SignOPT":"Sign-OPT", "SVMOPT":"SVM-OPT", "boundary_attack":"Boundary Attack"}
                        #, "RayS": "RayS","GeoDA": "GeoDA"}
                        #"biased_boundary_attack": "Biased Boundary Attack"}

def from_method_to_dir_path(dataset, method, norm, targeted):
    if method == "tangent_attack":
        path = "{method}_ablation_study_initial_sample-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                             norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "HSJA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                             norm=norm,  target_str="untargeted" if not targeted else "targeted_increment")
    if method == "tangent_attack@30":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "HSJA@30":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm,  target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "GeoDA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "biased_boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "RayS":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SignOPT":
        if targeted and dataset == "ImageNet":
            path = "{method}_random_start_point-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                                   target_str="untargeted" if not targeted else "targeted_increment")
        else:
            path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SVMOPT":
        if targeted and dataset == "ImageNet":
            path = "{method}_random_start_point-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                                   target_str="untargeted" if not targeted else "targeted_increment")
        else:
            path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    return path


def get_all_exists_folder(dataset, methods, norm, targeted):
    root_dir = "/home1/machen/hard_label_attacks/logs/"
    dataset_path_dict = {}  # dataset_path_dict {("CIFAR-10","l2","untargeted", "NES"): "/.../"， }
    for method in methods:
        file_name = from_method_to_dir_path(dataset, method, norm, targeted)
        file_path = root_dir + file_name
        if os.path.exists(file_path):
            dataset_path_dict[(dataset, norm, targeted, method_name_to_paper[method])] = file_path
        else:
            print("{} does not exist!!!".format(file_path))
    return dataset_path_dict

initial_target_sample_schedule_dict = {"random_initial_sample": "随机选择图片初始化", "best_initial_sample": "选择失真度最小的图片初始化",
                      "worst_initial_sample": "选择失真度最大的图片初始化"}

def draw_query_distortion_figure(dataset, norm, targeted, arch, fig_type, dump_file_path, xlabel, ylabel):

    # fig_type can be [query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query]
    methods = list(method_name_to_paper.keys())
    dataset_path_dict= get_all_exists_folder(dataset, methods, norm, targeted)
    max_query = 10000
    if dataset=="ImageNet" and targeted:
        max_query = 20000
    query_budgets = np.arange(1000, max_query+1, 1000)
    query_budgets = np.insert(query_budgets,0,500)
    # query_budgets = np.insert(query_budgets,0, [200,300,400])
    data_info = read_all_data(dataset_path_dict, arch, query_budgets, fig_type)  # fig_type can be mean_distortion or median_distortion
    plt.style.use('seaborn-whitegrid')
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['simsun']  # 显示中文标签
    rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置

    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['pgf.preamble'] = "\n".join([
        r"\usepackage{url}",  # load additional packages
        r"\usepackage{unicode-math}",  # unicode math setup
        r"\setmainfont{DejaVu Serif}",  # serif font via preamble
    ])


    plt.figure(figsize=(15, 15))
    colors = ['b', 'g', 'c', 'y', 'k', 'peru', "gold"]
    markers = ['o', '>', '*', 's', "X", "h"]
    linestyles = ["solid", "dashed", "densely dotted", "dashdotdotted", "densely dashed", "densely dashdotdotted"]

    our_schedule = 'random_initial_sample'

    xtick = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    if max_query == 20000:
        xtick = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000])
    max_y = 0
    min_y= 999
    for idx, ((dataset, norm, targeted, initial_schedule), (x,y)) in enumerate(data_info.items()):
        color = colors[idx%len(colors)]
        if initial_schedule == our_schedule:
            color = "r"
        x = np.asarray(x)
        y = np.asarray(y)
        if np.max(y) > max_y:
            max_y = np.max(y)
        if np.min(y) < min_y:
            min_y = np.min(y)
        line, = plt.plot(x, y, label=initial_target_sample_schedule_dict[initial_schedule], color=color,
                         linestyle=linestyle_dict[linestyles[idx]], marker=markers[idx],
                         markersize=20, linewidth=3.0)
    if dataset!="ImageNet":
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    else:
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    if dataset == "ImageNet" and targeted:
        plt.xlim(0, max_query+1000)
    else:
        plt.xlim(0, max_query)

    plt.ylim(0, max_y+0.1)
    plt.gcf().subplots_adjust(bottom=0.15)
    # xtick = [0, 5000, 10000]
    if dataset == "ImageNet" and targeted:
        x_ticks = xtick[::2]
        x_ticks = x_ticks.tolist()
        x_ticks_label = ["{}K".format(x_tick // 1000) for x_tick in x_ticks]
        x_ticks_label[0] = "0"
        plt.xticks(x_ticks, x_ticks_label, fontsize=45)  # remove 500
    else:
        x_ticks_label = ["{}K".format(x_tick // 1000) for x_tick in xtick[1:]]
        plt.xticks(xtick[1:],x_ticks_label, fontsize=45) # remove 500
    yticks = np.arange(0, max_y + 1, 5)
    plt.yticks(yticks[1:], fontsize=45)

    plt.xlabel(xlabel, fontsize=55)
    plt.ylabel(ylabel, fontsize=55)
    plt.legend(loc='upper right', prop={'size': 45},fancybox=True, framealpha=0.5,frameon=True)
    plt.savefig(dump_file_path, backend='pgf', dpi=300)
    plt.close()
    log.info("write file into {}".format(dump_file_path))



def parse_args():
    parser = argparse.ArgumentParser(description='Drawing Figures of Attacking Normal Models')
    parser.add_argument("--fig_type", type=str, choices=["mean_distortion",
                                                         "median_distortion"])
    parser.add_argument("--dataset", type=str, required=True, help="the dataset to train")
    parser.add_argument("--norm", type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument("--targeted", action="store_true", help="Does it train on the data of targeted attack?")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dump_folder = "/home1/machen/hard_label_attacks/paper_chinese_figures/initial_target_sample_ablation_study/"
    os.makedirs(dump_folder, exist_ok=True)

    arch = "resnet101"
    file_path  = dump_folder + "{dataset}_{model}_{norm}_{target_str}_attack_initial_sample_ablation_study.pdf".format(dataset=args.dataset,
                  model=arch, norm=args.norm, target_str="untargeted" if not args.targeted else "targeted")
    x_label = "查询预算次数"
    if args.fig_type == "mean_distortion":
        y_label = "平均$\ell_2$范数失真度"
    elif args.fig_type == "median_distortion":
        y_label = "$\ell_2$范数失真度的中位数"
    draw_query_distortion_figure(args.dataset, args.norm, args.targeted, arch, args.fig_type, file_path, x_label,y_label)
