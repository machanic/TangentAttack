import os
import sys
sys.path.append(os.getcwd())
import argparse
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.interpolate import make_interp_spline
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

from config import MODELS_TEST_STANDARD
from matplotlib import rcParams, rc

rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rc('pdf', fonttype=42)

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
    for (dataset, norm, targeted, method, num_eval_grad), dir_path in dataset_path_dict.items():
        for file_path in os.listdir(dir_path):
            if arch in file_path and file_path.endswith(".json") and not file_path.startswith("tmp"):
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
                data_info[(dataset, norm, targeted, method, num_eval_grad)] = (x,y)
                break
    return data_info




method_name_to_paper = {"tangent_attack":"TA",
                        "ellipsoid_tangent_attack":"G-TA",
                        "HSJA":"HSJA", "HSJARandom":"RandomHSJA"}
                       # "SignOPT":"Sign-OPT", "SVMOPT":"SVM-OPT"}
                        #"boundary_attack":"Boundary Attack", "RayS": "RayS","GeoDA": "GeoDA"}
                        #"biased_boundary_attack": "Biased Boundary Attack"}

def from_method_to_dir_path(dataset, method, norm, targeted, num_eval_grads):
    if method == "tangent_attack":
        path = "{method}@{num_eval_grads}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,num_eval_grads=num_eval_grads,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "HSJA":
        path = "{method}@{num_eval_grads}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, num_eval_grads=num_eval_grads,
                                                                norm=norm,  target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "HSJARandom":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, num_eval_grads=num_eval_grads,
                                                                norm=norm,  target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "GeoDA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "biased_boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "RayS":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SignOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SVMOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    return path

def from_method_to_dir_path_for_direction_study(dataset, method, norm, targeted, num_eval_grads):
    if method == "tangent_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,num_eval_grads=num_eval_grads,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    if method == "ellipsoid_tangent_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,num_eval_grads=num_eval_grads,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "HSJA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, num_eval_grads=num_eval_grads,
                                                                norm=norm,  target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "HSJARandom":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, num_eval_grads=num_eval_grads,
                                                                norm=norm,  target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "GeoDA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "biased_boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "RayS":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SignOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SVMOPT":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    return path

def get_all_exists_folder(dataset, method, norm, targeted, num_eval_grads):
    root_dir = "/home1/machen/hard_label_attacks/logs/"
    dataset_path_dict = {}  # dataset_path_dict {("CIFAR-10","l2","untargeted", "NES"): "/.../"， }
    for num_eval_grad in num_eval_grads:
        file_name = from_method_to_dir_path(dataset, method, norm, targeted, num_eval_grad)
        file_path = root_dir + file_name
        if os.path.exists(file_path):
            dataset_path_dict[(dataset, norm, targeted, method_name_to_paper[method], num_eval_grad)] = file_path
        else:
            print("{} does not exist".format(file_path))
    return dataset_path_dict

def get_all_exists_folder_multiple_methods(dataset, methods, norm, targeted, num_eval_grads):
    root_dir = "/home1/machen/hard_label_attacks/logs/"
    dataset_path_dict = {}  # dataset_path_dict {("CIFAR-10","l2","untargeted", "NES"): "/.../"， }
    for method in methods:
        file_name = from_method_to_dir_path_for_direction_study(dataset, method, norm, targeted, num_eval_grads[0])
        file_path = root_dir + file_name
        if os.path.exists(file_path):
            dataset_path_dict[(dataset, norm, targeted, method_name_to_paper[method],num_eval_grads[0])] = file_path
        else:
            print("{} does not exist!!!".format(file_path))
    return dataset_path_dict


def draw_query_distortion_figure(dataset, norm, targeted, arch, fig_type, dump_file_path, xlabel, ylabel):

    # fig_type can be [query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query]
    method = "tangent_attack"
    dataset_path_dict= get_all_exists_folder(dataset, method, norm, targeted, [5,10,30,50,70,100,150,200])
    max_query = 10000
    if dataset=="ImageNet" and targeted:
        max_query = 20000
    query_budgets = np.arange(1000, max_query+1, 1000)
    query_budgets = np.insert(query_budgets,0,500)
    # query_budgets = np.insert(query_budgets,0, [200,300,400])
    data_info = read_all_data(dataset_path_dict, arch, query_budgets, fig_type)  # fig_type can be mean_distortion or median_distortion
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(15, 15))
    colors = ['b', 'g', 'r', 'y', 'k', 'peru', "gold","rosybrown"]
    markers = ['o', '>', 'P', '*', "X", "h","s","D"]
    linestyles = ["solid", "dashed", "densely dotted", "dashdotdotted", "densely dashed", "densely dashdotdotted","loosely dashed","dashdot"]


    xtick = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    if max_query == 20000:
        xtick = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000])
    max_y = 0
    min_y= 999
    for idx, ((dataset, norm, targeted, method, num_eval_grad), (x,y)) in enumerate(data_info.items()):
        color = colors[idx]
        # if num_eval_grad==30:
        #     continue
        x = np.asarray(x)
        y = np.asarray(y)
        if np.max(y) > max_y:
            max_y = np.max(y)
        if np.min(y) < min_y:
            min_y = np.min(y)
        line, = plt.plot(x, y, label="$B_0={}$".format(num_eval_grad), color=color, linestyle=linestyle_dict[linestyles[idx]],
                         marker=markers[idx], markersize=20, linewidth=3.0)


    # for idx, ((dataset, norm, targeted, method, num_eval_grad), (x,y)) in enumerate(data_info.items()):
    #     color = colors[idx]
    #     if num_eval_grad!=30:
    #         continue
    #     if num_eval_grad==30:
    #         color='r'
    #     x = np.asarray(x)
    #     y = np.asarray(y)
    #     if np.max(y) > max_y:
    #         max_y = np.max(y)
    #     if np.min(y) < min_y:
    #         min_y = np.min(y)
    #     line, = plt.plot(x, y, label="$B_0={}$".format(num_eval_grad), color=color, linestyle=linestyle_dict[linestyles[idx]],
    #                      marker=markers[idx], markersize=20, linewidth=3.0)

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
    print("max y is {}".format(max_y))
    # xtick = [0, 5000, 10000]
    if dataset == "ImageNet" and targeted:
        x_ticks = xtick[::2]
        x_ticks = x_ticks.tolist()
        x_ticks_label = ["{}K".format(x_tick // 1000) for x_tick in x_ticks]
        x_ticks_label[0] = "0"
        x_ticks[0] = 0
        plt.xticks(x_ticks, x_ticks_label, fontsize=38)  # remove 500
    else:
        x_ticks_label = ["{}K".format(x_tick // 1000) for x_tick in xtick]
        xtick[0] = 0
        x_ticks_label[0] = "0"
        plt.xticks(xtick, x_ticks_label, fontsize=38) # remove 500
    if dataset=="ImageNet":
        yticks = np.arange(0, max_y+1, 5)
        plt.yticks(yticks, fontsize=38)
    else:
        plt.yticks([0.1,1, max_y/2, max_y+0.1], fontsize=38)
    plt.xlabel(xlabel, fontsize=55)
    plt.ylabel(ylabel, fontsize=55)
    plt.legend(loc='upper right', prop={'size': 45},labelcolor='linecolor',fancybox=True, framealpha=0.5,frameon=True)
    plt.savefig(dump_file_path, dpi=200)
    plt.close()
    print("save to {}".format(dump_file_path))

def draw_random_HSJA_TangentAttack_query_distortion_figure(dataset, norm, targeted, arch, fig_type, dump_file_path, xlabel, ylabel):

    # fig_type can be [query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query]
    methods = list(method_name_to_paper.keys())
    dataset_path_dict= get_all_exists_folder_multiple_methods(dataset, methods, norm, targeted, [100])
    max_query = 10000
    if dataset=="ImageNet" and targeted:
        max_query = 20000
    query_budgets = np.arange(1000, max_query+1, 1000)
    query_budgets = np.insert(query_budgets,0,500)
    # query_budgets = np.insert(query_budgets,0, [200,300,400])
    data_info = read_all_data(dataset_path_dict, arch, query_budgets, fig_type)  # fig_type can be mean_distortion or median_distortion
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(15, 15))
    colors = ['b', 'g', 'c', 'y', 'k', 'peru', "gold", "rosybrown"]
    markers = ['>', 'o', '*', "P", "+", "h"]
    linestyles = ["dashed","solid","densely dotted", "densely dashed", "densely dashdotdotted",
                  "loosely dashed", "dashdot"]
    our_method1 = 'TA'
    our_method2 = 'G-TA'

    xtick = np.array([500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    if max_query == 20000:
        xtick = np.array([500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000])
    max_y = 0
    min_y= 999
    for idx, ((dataset, norm, targeted, method, num_eval_grad), (x,y)) in enumerate(data_info.items()):
        color = colors[idx%len(colors)]
        if method == our_method2:
            color = "r"
        x = np.asarray(x)
        y = np.asarray(y)
        if np.max(y) > max_y:
            max_y = np.max(y)
        if np.min(y) < min_y:
            min_y = np.min(y)
        line, = plt.plot(x, y, label=method, color=color, linestyle=linestyle_dict[linestyles[idx]],
                         marker=markers[idx], markersize=20, linewidth=3.0)
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
    print("max y is {}".format(max_y))
    # xtick = [0, 5000, 10000]
    if dataset == "ImageNet" and targeted:
        x_ticks = xtick[::2]
        x_ticks = x_ticks.tolist()
        x_ticks_label = ["{}K".format(x_tick // 1000) for x_tick in x_ticks]
        x_ticks[0] = 0
        x_ticks_label[0] = "0"
        plt.xticks(x_ticks,x_ticks_label, fontsize=38)  # remove 500
    else:
        x_ticks_label = ["{}K".format(x_tick // 1000) for x_tick in xtick]
        xtick[0] = 0
        x_ticks_label[0] = "0"
        plt.xticks(xtick,x_ticks_label, fontsize=38) # remove 500
    if dataset=="ImageNet":
        yticks = np.arange(0, max_y+1, 5)
        plt.yticks(yticks, fontsize=38)
    else:
        plt.yticks([0.1,1, max_y/2, max_y+0.1], fontsize=38)
    plt.xlabel(xlabel, fontsize=55)
    plt.ylabel(ylabel, fontsize=55)
    plt.legend(loc='upper right', prop={'size': 45},labelcolor='linecolor',fancybox=True, framealpha=0.5,frameon=True)
    plt.savefig(dump_file_path, dpi=200)
    plt.close()
    print("save to {}".format(dump_file_path))


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
    dump_folder = "/home1/machen/hard_label_attacks/ablation_study_figures/{}/".format(args.fig_type)
    os.makedirs(dump_folder, exist_ok=True)

    if "CIFAR" in args.dataset:
        archs = ['pyramidnet272',"gdas","WRN-28-10-drop", "WRN-40-10-drop"]
    else:
        archs = ["resnet101"]

    for model in archs:
        file_path  = dump_folder + "{dataset}_{model}_{norm}_{target_str}_attack.pdf".format(dataset=args.dataset,
                      model=model, norm=args.norm, target_str="untargeted" if not args.targeted else "targeted")
        x_label = "Number of Queries"
        if args.fig_type == "mean_distortion":
            y_label = "Mean $\ell_2$ Distortion"
        elif args.fig_type == "median_distortion":
            y_label = "Median $\ell_2$ Distortion"

        draw_query_distortion_figure(args.dataset, args.norm, args.targeted, model, args.fig_type, file_path,x_label,y_label)
        # draw_random_HSJA_TangentAttack_query_distortion_figure(args.dataset, args.norm, args.targeted,
        #                                                         model, args.fig_type, file_path, x_label,y_label)
        # elif args.fig_type == "query_hist":
        #     target_str = "/untargeted" if not args.targeted else "targeted"
        #     os.makedirs(dump_folder, exist_ok=True)
        #     for dataset in ["CIFAR-10","CIFAR-100", "TinyImageNet"]:
        #         if "CIFAR" in dataset:
        #             archs = ['pyramidnet272', "gdas", "WRN-28-10-drop", "WRN-40-10-drop"]
        #         else:
        #             archs = ["densenet121", "resnext32_4", "resnext64_4"]
        #         for norm in ["l2","linf"]:
        #             for model in archs:
        #                 draw_histogram_fig(dataset, norm, args.targeted, model, dump_folder + target_str)
