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


def read_json_data(json_path):
    # data_key can be query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query
    print("begin read {}".format(json_path))
    with open(json_path, "r") as file_obj:
        data_txt = file_obj.read()
        data_json = json.loads(data_txt)
        cosine_grad_dict = data_json["approx_and_true_grad_cosine_similarity"]
        correct_all = np.array(data_json["correct_all"]).astype(np.bool)
        #success_all = np.array(data_json["success_all"]).astype(np.int32)
    return cosine_grad_dict,  correct_all

def read_all_data(dataset_path_dict, arch, query_budgets, stats="mean_cosine"):
    # dataset_path_dict {("CIFAR-10","l2","untargeted"): "/.../"， }
    data_info = {}
    for (dataset, norm, targeted, method), dir_path in dataset_path_dict.items():
        file_path = "{}_replace_with_true_gradient_result.json".format(arch)
        file_path = dir_path + "/" + file_path
        cosine_grad_dict, correct_all = read_json_data(file_path)
        x = []
        y = []
        for query_budget in query_budgets:
            cosine_gradient_similarity = []
            for image_id, query_cosine_dict in cosine_grad_dict.items():
                query_cosine_sim_dict = {int(float(query)): float(cosine) for query, cosine in query_cosine_dict.items()}
                queries = np.array(list(query_cosine_sim_dict.keys()))
                queries = np.sort(queries)
                find_index = np.searchsorted(queries, query_budget, side='right') - 1
                if query_budget < queries[find_index]:
                    print(
                        "query budget is {}, find query is {}, min query is {}, len query_distortion is {}".format(
                            query_budget, queries[find_index], np.min(queries).item(),
                            len(query_cosine_sim_dict)))
                    continue
                cosine_gradient_similarity.append(query_cosine_sim_dict[queries[find_index]])
            cosine_gradient_similarity = np.array(cosine_gradient_similarity)
            cosine_gradient_similarity = cosine_gradient_similarity[~np.isnan(cosine_gradient_similarity)]  # 去掉nan的值
            mean_cosine_grad = np.mean(cosine_gradient_similarity)
            median_cosine_grad = np.median(cosine_gradient_similarity)
            x.append(query_budget)
            if stats == "mean_cosine":
                y.append(mean_cosine_grad)
            elif stats == "median_cosine":
                y.append(median_cosine_grad)
        x = np.array(x)
        y = np.array(y)
        data_info[(dataset, norm, targeted, method)] = (x,y)
    return data_info




method_name_to_paper = {"tangent_attack":"Tangent Attack(hemisphere)"}


def from_method_to_dir_path(dataset, method, norm, targeted):
    if method == "tangent_attack":
        path = "{method}_ablation_study-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
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
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g',  'c', 'm', 'y', 'k', 'orange', "pink","brown","slategrey","cornflowerblue","greenyellow"]
    # markers = [".",",","o","^","s","p","x"]
    # max_x = 0
    # min_x = 0
    our_method = 'Tangent Attack(hemisphere)'

    xtick = np.array([500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    if max_query == 20000:
        xtick = np.array([500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000])
    max_y = 0
    min_y= 999
    for idx, ((dataset, norm, targeted, method), (x,y)) in enumerate(data_info.items()):
        color = colors[idx%len(colors)]
        if method == our_method:
            color = "r"

        x = np.asarray(x)
        y = np.asarray(y)
        if np.max(y) > max_y:
            max_y = np.max(y)
        if np.min(y) < min_y:
            min_y = np.min(y)
        line, = plt.plot(x, y, label=method, color=color, linestyle="-",linewidth=1)
        #line, = plt.plot(x, y, label=method, color=color, linestyle="-")
        y_points = np.interp(xtick, x, y)
        plt.scatter(xtick, y_points,color=color,marker='.',s=30)
        # plt.scatter(xtick, y_points, color=color, marker='.')
    if dataset!="ImageNet":
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    else:
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    if dataset == "ImageNet" and targeted:
        plt.xlim(0, max_query+1000)
    else:
        plt.xlim(0, max_query)

    # plt.ylim(0, max_y+0.1)
    plt.gcf().subplots_adjust(bottom=0.15)
    print("max y is {}".format(max_y))
    # xtick = [0, 5000, 10000]
    if dataset == "ImageNet" and targeted:
        x_ticks = xtick[1::2]
        x_ticks = x_ticks.tolist()
        x_ticks.append(21000)
        x_ticks_label = ["{}K".format(x_tick // 1000) for x_tick in x_ticks]
        plt.xticks(x_ticks, x_ticks_label, fontsize=18)  # remove 500
    else:
        x_ticks_label = ["{}K".format(x_tick // 1000) for x_tick in xtick[1:]]
        plt.xticks(xtick[1:],x_ticks_label, fontsize=18) # remove 500
    # if dataset=="ImageNet":
    #     yticks = np.arange(0, max_y+1, 5)
    #     plt.yticks(yticks, fontsize=18)
    # else:
    #     plt.yticks([0.1,1, max_y/2, max_y+0.1], fontsize=18)
    y_ticks_label = ["{:.3f}".format(e) for e in np.arange(0.0,max_y,0.005).tolist()]
    y_ticks_label[0]= "0"
    plt.yticks(np.arange(0.0,max_y,0.005), y_ticks_label, fontsize=18)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.legend(loc='upper right', prop={'size': 20})
    plt.savefig(dump_file_path, dpi=200)
    plt.close()
    print("save to {}".format(dump_file_path))



def parse_args():
    parser = argparse.ArgumentParser(description='Drawing Figures of Attacking Normal Models')
    parser.add_argument("--fig_type", type=str, choices=["mean_cosine",
                                                         "median_cosine"])
    parser.add_argument("--dataset", type=str, required=True, help="the dataset to train")
    parser.add_argument("--norm", type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument("--targeted", action="store_true", help="Does it train on the data of targeted attack?")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dump_folder = "/home1/machen/hard_label_attacks/paper_figures/rebuttal_R4/"
    os.makedirs(dump_folder, exist_ok=True)

    if "CIFAR" in args.dataset:
        archs = ["WRN-28-10-drop"]
    else:
        archs = ["resnet101"]
    for model in archs:
        file_path  = dump_folder + "true_gradient_cosine_{dataset}_{model}_{norm}_{target_str}_attack.pdf".format(dataset=args.dataset,
                      model=model, norm=args.norm, target_str="untargeted" if not args.targeted else "targeted")
        x_label = "Number of Queries"
        if args.fig_type == "mean_cosine":
            y_label = "Avg. Cosine Similarity"
        elif args.fig_type == "median_cosine":
            y_label = "Median Cosine Similarity"

        draw_query_distortion_figure(args.dataset, args.norm, args.targeted, model, args.fig_type, file_path,x_label,y_label)

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
