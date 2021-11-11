import os
import sys
sys.path.append(os.getcwd())
import argparse
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.interpolate import make_interp_spline
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

from config import MODELS_TEST_STANDARD

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


method_name_to_paper = {"tangent_attack":"TA",
                        "ellipsoid_tangent_attack":"G-TA",
                        "HSJA":"HSJA"}

def read_json_data(json_path):
    # data_key can be query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query
    print("begin read {}".format(json_path))
    with open(json_path, "r") as file_obj:
        data_txt = file_obj.read()
        data_json = json.loads(data_txt)
        distortion_dict = data_json["distortion"]
        correct_all = np.array(data_json["correct_all"]).astype(np.bool)
    return distortion_dict,  correct_all

def read_all_data(dataset_path_dict, arch, query_budgets):
    data_info = defaultdict(dict)  # 最外层是query_budget
    for (dataset, norm, targeted, method), dir_path in dataset_path_dict.items():
        for file_path in os.listdir(dir_path):
            if arch in file_path and file_path.endswith(".json") and not file_path.startswith("tmp"):
                file_path = dir_path + "/" + file_path
                distortion_dict, correct_all = read_json_data(file_path)
                for query_budget in query_budgets:
                    distortions = defaultdict(list)
                    for image_id, query_distortion_dict in distortion_dict.items():
                        query_distortion_dict = {int(float(query)): float(dist) for query, dist in query_distortion_dict.items()}
                        queries = np.array(list(query_distortion_dict.keys()))
                        queries = np.sort(queries)
                        find_index = np.searchsorted(queries, query_budget, side='right') - 1
                        assert queries[find_index] <= query_budget
                        if query_budget < queries[find_index]:
                            print(
                                "query budget is {}, find query is {}, min query is {}, len query_distortion is {}".format(
                                    query_budget, queries[find_index], np.min(queries).item(),
                                    len(query_distortion_dict)))
                            continue
                        distortions[int(image_id)].append(query_distortion_dict[queries[find_index]])
                    distortion_list = [entry[1] for entry in sorted(distortions.items(),key=lambda e:e[0])]
                    distortion_list = np.array(distortion_list)
                    distortion_list = distortion_list[~np.isnan(distortion_list)]  # 去掉nan的值
                    assert len(distortion_list) == 1000
                    data_info[query_budget][(dataset, norm, targeted, method)] = distortion_list
                break
    return data_info

def from_method_to_dir_path(dataset, method, norm, targeted):
    if method == "tangent_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    if method == "ellipsoid_tangent_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "HSJA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm,  target_str="untargeted" if not targeted else "targeted_increment")
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


def draw_query_distortion_figure(dataset, norm, targeted, arch, dump_folder_path, xlabel, ylabel):

    methods = list(method_name_to_paper.keys())
    dataset_path_dict= get_all_exists_folder(dataset, methods, norm, targeted)
    max_query = 10000
    if dataset=="ImageNet" and targeted:
        max_query = 20000
    query_budgets = np.arange(1000, max_query+1, 1000)
    data_info = read_all_data(dataset_path_dict, arch, query_budgets)  # fig_type can be mean_distortion or median_distortion
    colordict = {'G-TA':'r','TA':'g',"HSJA":'c'}
    markers = ['o', '>', 'X', 's', "D", "h", "P", "*","1","p"]
    linestyles = ["solid", "dashed", "densely dotted", "dashdotdotted", "densely dashed", "densely dashdotdotted",
                  "loosely dashed", "dashdot","dotted","loosely dashed"]
    xtick = np.asarray(np.arange(1000)[::100].tolist() + [1000])
    xtick_label = [x // 100 for x in xtick]
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
    max_y  = 0
    for query_budget, dist_dict in data_info.items():
        plt.figure(figsize=(16, 16))
        plt.gcf().subplots_adjust(bottom=0.15)
        idx = 0
        for (dataset, norm, targeted, method), distortion_list in dist_dict.items():
            color = colordict[method]
            y = np.asarray(distortion_list[::50].tolist() + [distortion_list[-1].item()])
            x = np.asarray(np.arange(len(distortion_list))[::50].tolist() + [1000])
            if np.max(y) > max_y:
                max_y = np.max(y)
            line, = plt.plot(x, y, label=method,
                             color=color, linestyle=linestyle_dict[linestyles[idx]],
                             marker=markers[idx], markersize=20, linewidth=3.0)
            idx+=1
        plt.xlabel(xlabel, fontsize=55)
        plt.ylabel(ylabel, fontsize=55)

        plt.xticks(xtick, xtick_label, fontsize=40)  # remove 500
        plt.yticks(np.arange(0, max_y, 50), fontsize=40)
        plt.legend(loc='upper right', prop={'size': 45},fancybox=True, framealpha=0.5,frameon=True)
        target_str = "untargeted" if not targeted else "targeted"
        dump_file_path = dump_folder_path + '/{}/{}_max_query_{}_distribution.pdf'.format(arch, target_str, query_budget)
        os.makedirs(os.path.dirname(dump_file_path),exist_ok=True)
        plt.savefig(dump_file_path, backend='pgf',dpi=300)
        plt.clf()
        plt.close('all')
        print("save to {}".format(dump_file_path))

def parse_args():
    parser = argparse.ArgumentParser(description='Drawing Figures of Attacking Normal Models')
    parser.add_argument("--dataset", type=str, required=True, help="the dataset to train")
    parser.add_argument("--norm", type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument("--targeted", action="store_true", help="Does it train on the data of targeted attack?")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dump_folder = "/home1/machen/hard_label_attacks/paper_chinese_figures/distortion_individual_image_distribution/{}".format(args.dataset)
    os.makedirs(dump_folder, exist_ok=True)

    if "CIFAR" in args.dataset:
        archs = ['pyramidnet272',"gdas","WRN-28-10-drop", "WRN-40-10-drop"]
    else:
        archs = ["resnet101","inceptionv4","senet154","inceptionv3","resnext101_64x4d"]


    for model in archs:
        x_label = "图片编号"
        if args.norm == "l2":
            y_label = "$\ell_2$范数失真度"
        else:
            y_label = "$\ell_\infty$范数失真度"
        draw_query_distortion_figure(args.dataset, args.norm, args.targeted, model, dump_folder, x_label, y_label)
