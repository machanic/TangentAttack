from collections import defaultdict

import bisect
import numpy as np
import json
import os

def new_round(_float, _len):
    """
    Parameters
    ----------
    _float: float
    _len: int, 指定四舍五入需要保留的小数点后几位数为_len

    Returns
    -------
    type ==> float, 返回四舍五入后的值
    """
    if isinstance(_float, float):
        if str(_float)[::-1].find('.') <= _len:
            return (_float)
        if str(_float)[-1] == '5':
            return (round(float(str(_float)[:-1] + '6'), _len))
        else:
            return (round(_float, _len))
    else:
        return (round(_float, _len))


method_name_to_paper = {"tangent_attack":"Tangent Attack",  "HSJA":"HopSkipJumpAttack",
                        "SignOPT":"Sign-OPT", "SVMOPT":"SVM-OPT",
                        "RayS": "RayS","GeoDA": "GeoDA"}
                        #"biased_boundary_attack": "Biased Boundary Attack"}

def from_method_to_dir_path(dataset, method, norm, targeted):
    if method == "tangent_attack":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "HSJA":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm,  target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "GeoDA":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "biased_boundary_attack":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "RayS":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SignOPT":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SVMOPT":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    return path


def read_json_and_extract(json_path):
    with open(json_path, "r") as file_obj:
        json_content = json.load(file_obj)
        distortion = json_content["distortion"]
        return distortion

def get_file_name_list(dataset, method_name_to_paper, norm, targeted):
    folder_path_dict = {}
    for method, paper_method_name in method_name_to_paper.items():
        file_path = "/home1/machen/hard_label_attacks/logs/" + from_method_to_dir_path(dataset, method, norm, targeted)
        folder_path_dict[paper_method_name] = file_path
    return folder_path_dict

def bin_search(arr, target):
    if target not in arr:
        return None
    arr.sort()
    return arr[arr.index(target)-1], arr.index(target)-1


def get_mean_and_median_distortion_given_query_budgets(distortion_dict, query_budgets, want_key):
    mean_and_median_distortions = {}
    for query_budget in query_budgets:
        distortion_list = []
        for image_index, query_distortion in distortion_dict.items():
            query_distortion = {float(query):float(dist) for query,dist in query_distortion.items()}
            queries = list(query_distortion.keys())
            queries = np.sort(queries)
            find_index = bisect.bisect(queries, query_budget) - 1
            # print(len(queries),find_index)
            # find_index = np.searchsorted(queries, query_budget, side='right') - 1
            if query_budget < queries[find_index]:
                print("query budget is {}, find query is {}, min query is {}, len query_distortion is {}".format(query_budget, queries[find_index], np.min(queries).item(), len(query_distortion)))
                continue
            distortion_list.append(query_distortion[queries[find_index]])
        distortion_list = np.array(distortion_list)
        distortion_list = distortion_list[~np.isnan(distortion_list)]  # 去掉nan的值
        mean_distortion = np.mean(distortion_list)
        median_distortion = np.median(distortion_list)
        if want_key == "mean_distortion":
            mean_and_median_distortions[query_budget] = "{:.3f}".format(new_round(mean_distortion.item(),3))
        elif want_key =="median_distortion":
            mean_and_median_distortions[query_budget] = "{:.3f}".format(new_round(median_distortion.item(),3))
    return mean_and_median_distortions


def fetch_all_json_content_given_contraint(dataset, norm, targeted, arch, query_budgets, want_key="mean_distortion"):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm, targeted)
    result = {}
    for method, folder in folder_list.items():
        file_path = folder + "/resnet-50_{}_result.json".format(arch)
        if method in ["RayS","GeoDA"] and targeted:
            print("{} does not exist!".format(file_path))
            result[method] = defaultdict(lambda : "-")
            continue
        distortion_dict = read_json_and_extract(file_path)
        print(file_path)
        mean_and_median_distortions = get_mean_and_median_distortion_given_query_budgets(distortion_dict, query_budgets,want_key)
        result[method] = mean_and_median_distortions
    return result


def draw_wide_table_CIFAR(untargeted_result, targeted_result):
    print("""
                    & GeoDA & {untargeted_GeoDA_jpeg_1000} & {untargeted_GeoDA_jpeg_2000} & {untargeted_GeoDA_jpeg_5000} & {untargeted_GeoDA_jpeg_8000} & {untargeted_GeoDA_jpeg_10000} & {targeted_GeoDA_jpeg_1000} & {targeted_GeoDA_jpeg_2000} & {targeted_GeoDA_jpeg_5000} & {targeted_GeoDA_jpeg_8000} & {targeted_GeoDA_jpeg_10000}  \\\\
                    & RayS & {untargeted_RayS_jpeg_1000} & {untargeted_RayS_jpeg_2000} & {untargeted_RayS_jpeg_5000} & {untargeted_RayS_jpeg_8000} & {untargeted_RayS_jpeg_10000} & {targeted_RayS_jpeg_1000} & {targeted_RayS_jpeg_2000} & {targeted_RayS_jpeg_5000} & {targeted_RayS_jpeg_8000} & {targeted_RayS_jpeg_10000}  \\\\
                    & Sign-OPT & {untargeted_SignOPT_jpeg_1000} & {untargeted_SignOPT_jpeg_2000} & {untargeted_SignOPT_jpeg_5000} & {untargeted_SignOPT_jpeg_8000} & {untargeted_SignOPT_jpeg_10000} & {targeted_SignOPT_jpeg_1000} & {targeted_SignOPT_jpeg_2000} & {targeted_SignOPT_jpeg_5000} & {targeted_SignOPT_jpeg_8000} & {targeted_SignOPT_jpeg_10000}  \\\\
                    & SVM-OPT & {untargeted_SVMOPT_jpeg_1000} & {untargeted_SVMOPT_jpeg_2000} & {untargeted_SVMOPT_jpeg_5000} & {untargeted_SVMOPT_jpeg_8000} & {untargeted_SVMOPT_jpeg_10000} & {targeted_SVMOPT_jpeg_1000} & {targeted_SVMOPT_jpeg_2000} & {targeted_SVMOPT_jpeg_5000} & {targeted_SVMOPT_jpeg_8000} & {targeted_SVMOPT_jpeg_10000}  \\\\
                    & HopSkipJumpAttack & {untargeted_HSJA_jpeg_1000} & {untargeted_HSJA_jpeg_2000} & {untargeted_HSJA_jpeg_5000} & {untargeted_HSJA_jpeg_8000} & {untargeted_HSJA_jpeg_10000} & {targeted_HSJA_jpeg_1000} & {targeted_HSJA_jpeg_2000} & {targeted_HSJA_jpeg_5000} & {targeted_HSJA_jpeg_8000} & {targeted_HSJA_jpeg_10000}  \\\\
                    & Tangent Attack (ours) & {untargeted_Tangent_jpeg_1000} & {untargeted_Tangent_jpeg_2000} & {untargeted_Tangent_jpeg_5000} & {untargeted_Tangent_jpeg_8000} & {untargeted_Tangent_jpeg_10000} & {targeted_Tangent_jpeg_1000} & {targeted_Tangent_jpeg_2000} & {targeted_Tangent_jpeg_5000} & {targeted_Tangent_jpeg_8000} & {targeted_Tangent_jpeg_10000}  \\\\
                    \\midrule
                    & GeoDA & {untargeted_GeoDA_feature_distillation_1000} & {untargeted_GeoDA_feature_distillation_2000} & {untargeted_GeoDA_feature_distillation_5000} & {untargeted_GeoDA_feature_distillation_8000} & {untargeted_GeoDA_feature_distillation_10000} & {targeted_GeoDA_feature_distillation_1000} & {targeted_GeoDA_feature_distillation_2000} & {targeted_GeoDA_feature_distillation_5000} & {targeted_GeoDA_feature_distillation_8000} & {targeted_GeoDA_feature_distillation_10000}  \\\\
                    & RayS & {untargeted_RayS_feature_distillation_1000} & {untargeted_RayS_feature_distillation_2000} & {untargeted_RayS_feature_distillation_5000} & {untargeted_RayS_feature_distillation_8000} & {untargeted_RayS_feature_distillation_10000} & {targeted_RayS_feature_distillation_1000} & {targeted_RayS_feature_distillation_2000} & {targeted_RayS_feature_distillation_5000} & {targeted_RayS_feature_distillation_8000} & {targeted_RayS_feature_distillation_10000}  \\\\
                    & Sign-OPT & {untargeted_SignOPT_feature_distillation_1000} & {untargeted_SignOPT_feature_distillation_2000} & {untargeted_SignOPT_feature_distillation_5000} & {untargeted_SignOPT_feature_distillation_8000} & {untargeted_SignOPT_feature_distillation_10000} & {targeted_SignOPT_feature_distillation_1000} & {targeted_SignOPT_feature_distillation_2000} & {targeted_SignOPT_feature_distillation_5000} & {targeted_SignOPT_feature_distillation_8000} & {targeted_SignOPT_feature_distillation_10000}  \\\\
                    & SVM-OPT & {untargeted_SVMOPT_feature_distillation_1000} & {untargeted_SVMOPT_feature_distillation_2000} & {untargeted_SVMOPT_feature_distillation_5000} & {untargeted_SVMOPT_feature_distillation_8000} & {untargeted_SVMOPT_feature_distillation_10000} & {targeted_SVMOPT_feature_distillation_1000} & {targeted_SVMOPT_feature_distillation_2000} & {targeted_SVMOPT_feature_distillation_5000} & {targeted_SVMOPT_feature_distillation_8000} & {targeted_SVMOPT_feature_distillation_10000}  \\\\
                    & HopSkipJumpAttack & {untargeted_HSJA_feature_distillation_1000} & {untargeted_HSJA_feature_distillation_2000} & {untargeted_HSJA_feature_distillation_5000} & {untargeted_HSJA_feature_distillation_8000} & {untargeted_HSJA_feature_distillation_10000} & {targeted_HSJA_feature_distillation_1000} & {targeted_HSJA_feature_distillation_2000} & {targeted_HSJA_feature_distillation_5000} & {targeted_HSJA_feature_distillation_8000} & {targeted_HSJA_feature_distillation_10000}  \\\\
                    & Tangent Attack (ours) & {untargeted_Tangent_feature_distillation_1000} & {untargeted_Tangent_feature_distillation_2000} & {untargeted_Tangent_feature_distillation_5000} & {untargeted_Tangent_feature_distillation_8000} & {untargeted_Tangent_feature_distillation_10000} & {targeted_Tangent_feature_distillation_1000} & {targeted_Tangent_feature_distillation_2000} & {targeted_Tangent_feature_distillation_5000} & {targeted_Tangent_feature_distillation_8000} & {targeted_Tangent_feature_distillation_10000}  \\\\
                    \\midrule
                    & GeoDA & {untargeted_GeoDA_TRADES_1000} & {untargeted_GeoDA_TRADES_2000} & {untargeted_GeoDA_TRADES_5000} & {untargeted_GeoDA_TRADES_8000} & {untargeted_GeoDA_TRADES_10000} & {targeted_GeoDA_TRADES_1000} & {targeted_GeoDA_TRADES_2000} & {targeted_GeoDA_TRADES_5000} & {targeted_GeoDA_TRADES_8000} & {targeted_GeoDA_TRADES_10000}  \\\\
                    & RayS & {untargeted_RayS_TRADES_1000} & {untargeted_RayS_TRADES_2000} & {untargeted_RayS_TRADES_5000} & {untargeted_RayS_TRADES_8000} & {untargeted_RayS_TRADES_10000} & {targeted_RayS_TRADES_1000} & {targeted_RayS_TRADES_2000} & {targeted_RayS_TRADES_5000} & {targeted_RayS_TRADES_8000} & {targeted_RayS_TRADES_10000}  \\\\
                    & Sign-OPT & {untargeted_SignOPT_TRADES_1000} & {untargeted_SignOPT_TRADES_2000} & {untargeted_SignOPT_TRADES_5000} & {untargeted_SignOPT_TRADES_8000} & {untargeted_SignOPT_TRADES_10000} & {targeted_SignOPT_TRADES_1000} & {targeted_SignOPT_TRADES_2000} & {targeted_SignOPT_TRADES_5000} & {targeted_SignOPT_TRADES_8000} & {targeted_SignOPT_TRADES_10000}  \\\\
                    & SVM-OPT & {untargeted_SVMOPT_TRADES_1000} & {untargeted_SVMOPT_TRADES_2000} & {untargeted_SVMOPT_TRADES_5000} & {untargeted_SVMOPT_TRADES_8000} & {untargeted_SVMOPT_TRADES_10000} & {targeted_SVMOPT_TRADES_1000} & {targeted_SVMOPT_TRADES_2000} & {targeted_SVMOPT_TRADES_5000} & {targeted_SVMOPT_TRADES_8000} & {targeted_SVMOPT_TRADES_10000}  \\\\
                    & HopSkipJumpAttack & {untargeted_HSJA_TRADES_1000} & {untargeted_HSJA_TRADES_2000} & {untargeted_HSJA_TRADES_5000} & {untargeted_HSJA_TRADES_8000} & {untargeted_HSJA_TRADES_10000} & {targeted_HSJA_TRADES_1000} & {targeted_HSJA_TRADES_2000} & {targeted_HSJA_TRADES_5000} & {targeted_HSJA_TRADES_8000} & {targeted_HSJA_TRADES_10000}  \\\\
                    & Tangent Attack (ours) & {untargeted_Tangent_TRADES_1000} & {untargeted_Tangent_TRADES_2000} & {untargeted_Tangent_TRADES_5000} & {untargeted_Tangent_TRADES_8000} & {untargeted_Tangent_TRADES_10000} & {targeted_Tangent_TRADES_1000} & {targeted_Tangent_TRADES_2000} & {targeted_Tangent_TRADES_5000} & {targeted_Tangent_TRADES_8000} & {targeted_Tangent_TRADES_10000}  \\\\
                    \\midrule
                    & GeoDA & {untargeted_GeoDA_feature_scatter_1000} & {untargeted_GeoDA_feature_scatter_2000} & {untargeted_GeoDA_feature_scatter_5000} & {untargeted_GeoDA_feature_scatter_8000} & {untargeted_GeoDA_feature_scatter_10000} & {targeted_GeoDA_feature_scatter_1000} & {targeted_GeoDA_feature_scatter_2000} & {targeted_GeoDA_feature_scatter_5000} & {targeted_GeoDA_feature_scatter_8000} & {targeted_GeoDA_feature_scatter_10000}  \\\\
                    & RayS & {untargeted_RayS_feature_scatter_1000} & {untargeted_RayS_feature_scatter_2000} & {untargeted_RayS_feature_scatter_5000} & {untargeted_RayS_feature_scatter_8000} & {untargeted_RayS_feature_scatter_10000} & {targeted_RayS_feature_scatter_1000} & {targeted_RayS_feature_scatter_2000} & {targeted_RayS_feature_scatter_5000} & {targeted_RayS_feature_scatter_8000} & {targeted_RayS_feature_scatter_10000}  \\\\
                    & Sign-OPT & {untargeted_SignOPT_feature_scatter_1000} & {untargeted_SignOPT_feature_scatter_2000} & {untargeted_SignOPT_feature_scatter_5000} & {untargeted_SignOPT_feature_scatter_8000} & {untargeted_SignOPT_feature_scatter_10000} & {targeted_SignOPT_feature_scatter_1000} & {targeted_SignOPT_feature_scatter_2000} & {targeted_SignOPT_feature_scatter_5000} & {targeted_SignOPT_feature_scatter_8000} & {targeted_SignOPT_feature_scatter_10000}  \\\\
                    & SVM-OPT & {untargeted_SVMOPT_feature_scatter_1000} & {untargeted_SVMOPT_feature_scatter_2000} & {untargeted_SVMOPT_feature_scatter_5000} & {untargeted_SVMOPT_feature_scatter_8000} & {untargeted_SVMOPT_feature_scatter_10000} & {targeted_SVMOPT_feature_scatter_1000} & {targeted_SVMOPT_feature_scatter_2000} & {targeted_SVMOPT_feature_scatter_5000} & {targeted_SVMOPT_feature_scatter_8000} & {targeted_SVMOPT_feature_scatter_10000}  \\\\
                    & HopSkipJumpAttack & {untargeted_HSJA_feature_scatter_1000} & {untargeted_HSJA_feature_scatter_2000} & {untargeted_HSJA_feature_scatter_5000} & {untargeted_HSJA_feature_scatter_8000} & {untargeted_HSJA_feature_scatter_10000} & {targeted_HSJA_feature_scatter_1000} & {targeted_HSJA_feature_scatter_2000} & {targeted_HSJA_feature_scatter_5000} & {targeted_HSJA_feature_scatter_8000} & {targeted_HSJA_feature_scatter_10000}  \\\\
                    & Tangent Attack (ours) & {untargeted_Tangent_feature_scatter_1000} & {untargeted_Tangent_feature_scatter_2000} & {untargeted_Tangent_feature_scatter_5000} & {untargeted_Tangent_feature_scatter_8000} & {untargeted_Tangent_feature_scatter_10000} & {targeted_Tangent_feature_scatter_1000} & {targeted_Tangent_feature_scatter_2000} & {targeted_Tangent_feature_scatter_5000} & {targeted_Tangent_feature_scatter_8000} & {targeted_Tangent_feature_scatter_10000}  \\\\
	        \\midrule
	     & GeoDA & {untargeted_GeoDA_com_defend_1000} & {untargeted_GeoDA_com_defend_2000} & {untargeted_GeoDA_com_defend_5000} & {untargeted_GeoDA_com_defend_8000} & {untargeted_GeoDA_com_defend_10000} & {targeted_GeoDA_com_defend_1000} & {targeted_GeoDA_com_defend_2000} & {targeted_GeoDA_com_defend_5000} & {targeted_GeoDA_com_defend_8000} & {targeted_GeoDA_com_defend_10000}  \\\\
                    & RayS & {untargeted_RayS_com_defend_1000} & {untargeted_RayS_com_defend_2000} & {untargeted_RayS_com_defend_5000} & {untargeted_RayS_com_defend_8000} & {untargeted_RayS_com_defend_10000} & {targeted_RayS_com_defend_1000} & {targeted_RayS_com_defend_2000} & {targeted_RayS_com_defend_5000} & {targeted_RayS_com_defend_8000} & {targeted_RayS_com_defend_10000}  \\\\
                    & Sign-OPT & {untargeted_SignOPT_com_defend_1000} & {untargeted_SignOPT_com_defend_2000} & {untargeted_SignOPT_com_defend_5000} & {untargeted_SignOPT_com_defend_8000} & {untargeted_SignOPT_com_defend_10000} & {targeted_SignOPT_com_defend_1000} & {targeted_SignOPT_com_defend_2000} & {targeted_SignOPT_com_defend_5000} & {targeted_SignOPT_com_defend_8000} & {targeted_SignOPT_com_defend_10000}  \\\\
                    & SVM-OPT & {untargeted_SVMOPT_com_defend_1000} & {untargeted_SVMOPT_com_defend_2000} & {untargeted_SVMOPT_com_defend_5000} & {untargeted_SVMOPT_com_defend_8000} & {untargeted_SVMOPT_com_defend_10000} & {targeted_SVMOPT_com_defend_1000} & {targeted_SVMOPT_com_defend_2000} & {targeted_SVMOPT_com_defend_5000} & {targeted_SVMOPT_com_defend_8000} & {targeted_SVMOPT_com_defend_10000}  \\\\
                    & HopSkipJumpAttack & {untargeted_HSJA_com_defend_1000} & {untargeted_HSJA_com_defend_2000} & {untargeted_HSJA_com_defend_5000} & {untargeted_HSJA_com_defend_8000} & {untargeted_HSJA_com_defend_10000} & {targeted_HSJA_com_defend_1000} & {targeted_HSJA_com_defend_2000} & {targeted_HSJA_com_defend_5000} & {targeted_HSJA_com_defend_8000} & {targeted_HSJA_com_defend_10000}  \\\\
                    & Tangent Attack (ours) & {untargeted_Tangent_com_defend_1000} & {untargeted_Tangent_com_defend_2000} & {untargeted_Tangent_com_defend_5000} & {untargeted_Tangent_com_defend_8000} & {untargeted_Tangent_com_defend_10000} & {targeted_Tangent_com_defend_1000} & {targeted_Tangent_com_defend_2000} & {targeted_Tangent_com_defend_5000} & {targeted_Tangent_com_defend_8000} & {targeted_Tangent_com_defend_10000}  \\\\
                        """.format(

        untargeted_SignOPT_jpeg_1000=untargeted_result["jpeg"]["Sign-OPT"][1000],
        untargeted_SignOPT_jpeg_2000=untargeted_result["jpeg"]["Sign-OPT"][2000],
        untargeted_SignOPT_jpeg_5000=untargeted_result["jpeg"]["Sign-OPT"][5000],
        untargeted_SignOPT_jpeg_8000=untargeted_result["jpeg"]["Sign-OPT"][8000],
        untargeted_SignOPT_jpeg_10000=untargeted_result["jpeg"]["Sign-OPT"][10000],

        targeted_SignOPT_jpeg_1000=targeted_result["jpeg"]["Sign-OPT"][1000],
        targeted_SignOPT_jpeg_2000=targeted_result["jpeg"]["Sign-OPT"][2000],
        targeted_SignOPT_jpeg_5000=targeted_result["jpeg"]["Sign-OPT"][5000],
        targeted_SignOPT_jpeg_8000=targeted_result["jpeg"]["Sign-OPT"][8000],
        targeted_SignOPT_jpeg_10000=targeted_result["jpeg"]["Sign-OPT"][10000],

        untargeted_SignOPT_feature_distillation_1000=untargeted_result["feature_distillation"]["Sign-OPT"][1000],
        untargeted_SignOPT_feature_distillation_2000=untargeted_result["feature_distillation"]["Sign-OPT"][2000],
        untargeted_SignOPT_feature_distillation_5000=untargeted_result["feature_distillation"]["Sign-OPT"][5000],
        untargeted_SignOPT_feature_distillation_8000=untargeted_result["feature_distillation"]["Sign-OPT"][8000],
        untargeted_SignOPT_feature_distillation_10000=untargeted_result["feature_distillation"]["Sign-OPT"][10000],

        targeted_SignOPT_feature_distillation_1000=targeted_result["feature_distillation"]["Sign-OPT"][1000],
        targeted_SignOPT_feature_distillation_2000=targeted_result["feature_distillation"]["Sign-OPT"][2000],
        targeted_SignOPT_feature_distillation_5000=targeted_result["feature_distillation"]["Sign-OPT"][5000],
        targeted_SignOPT_feature_distillation_8000=targeted_result["feature_distillation"]["Sign-OPT"][8000],
        targeted_SignOPT_feature_distillation_10000=targeted_result["feature_distillation"]["Sign-OPT"][10000],

        untargeted_SignOPT_TRADES_1000=untargeted_result["TRADES"]["Sign-OPT"][1000],
        untargeted_SignOPT_TRADES_2000=untargeted_result["TRADES"]["Sign-OPT"][2000],
        untargeted_SignOPT_TRADES_5000=untargeted_result["TRADES"]["Sign-OPT"][5000],
        untargeted_SignOPT_TRADES_8000=untargeted_result["TRADES"]["Sign-OPT"][8000],
        untargeted_SignOPT_TRADES_10000=untargeted_result["TRADES"]["Sign-OPT"][10000],

        targeted_SignOPT_TRADES_1000=targeted_result["TRADES"]["Sign-OPT"][1000],
        targeted_SignOPT_TRADES_2000=targeted_result["TRADES"]["Sign-OPT"][2000],
        targeted_SignOPT_TRADES_5000=targeted_result["TRADES"]["Sign-OPT"][5000],
        targeted_SignOPT_TRADES_8000=targeted_result["TRADES"]["Sign-OPT"][8000],
        targeted_SignOPT_TRADES_10000=targeted_result["TRADES"]["Sign-OPT"][10000],

        untargeted_SignOPT_feature_scatter_1000=untargeted_result["feature_scatter"]["Sign-OPT"][1000],
        untargeted_SignOPT_feature_scatter_2000=untargeted_result["feature_scatter"]["Sign-OPT"][2000],
        untargeted_SignOPT_feature_scatter_5000=untargeted_result["feature_scatter"]["Sign-OPT"][5000],
        untargeted_SignOPT_feature_scatter_8000=untargeted_result["feature_scatter"]["Sign-OPT"][8000],
        untargeted_SignOPT_feature_scatter_10000=untargeted_result["feature_scatter"]["Sign-OPT"][10000],

        targeted_SignOPT_feature_scatter_1000=targeted_result["feature_scatter"]["Sign-OPT"][1000],
        targeted_SignOPT_feature_scatter_2000=targeted_result["feature_scatter"]["Sign-OPT"][2000],
        targeted_SignOPT_feature_scatter_5000=targeted_result["feature_scatter"]["Sign-OPT"][5000],
        targeted_SignOPT_feature_scatter_8000=targeted_result["feature_scatter"]["Sign-OPT"][8000],
        targeted_SignOPT_feature_scatter_10000=targeted_result["feature_scatter"]["Sign-OPT"][10000],

        untargeted_SignOPT_com_defend_1000=untargeted_result["com_defend"]["Sign-OPT"][1000],
        untargeted_SignOPT_com_defend_2000=untargeted_result["com_defend"]["Sign-OPT"][2000],
        untargeted_SignOPT_com_defend_5000=untargeted_result["com_defend"]["Sign-OPT"][5000],
        untargeted_SignOPT_com_defend_8000=untargeted_result["com_defend"]["Sign-OPT"][8000],
        untargeted_SignOPT_com_defend_10000=untargeted_result["com_defend"]["Sign-OPT"][10000],

        targeted_SignOPT_com_defend_1000=targeted_result["com_defend"]["Sign-OPT"][1000],
        targeted_SignOPT_com_defend_2000=targeted_result["com_defend"]["Sign-OPT"][2000],
        targeted_SignOPT_com_defend_5000=targeted_result["com_defend"]["Sign-OPT"][5000],
        targeted_SignOPT_com_defend_8000=targeted_result["com_defend"]["Sign-OPT"][8000],
        targeted_SignOPT_com_defend_10000=targeted_result["com_defend"]["Sign-OPT"][10000],

        untargeted_SVMOPT_jpeg_1000=untargeted_result["jpeg"]["SVM-OPT"][1000],
        untargeted_SVMOPT_jpeg_2000=untargeted_result["jpeg"]["SVM-OPT"][2000],
        untargeted_SVMOPT_jpeg_5000=untargeted_result["jpeg"]["SVM-OPT"][5000],
        untargeted_SVMOPT_jpeg_8000=untargeted_result["jpeg"]["SVM-OPT"][8000],
        untargeted_SVMOPT_jpeg_10000=untargeted_result["jpeg"]["SVM-OPT"][10000],

        targeted_SVMOPT_jpeg_1000=targeted_result["jpeg"]["SVM-OPT"][1000],
        targeted_SVMOPT_jpeg_2000=targeted_result["jpeg"]["SVM-OPT"][2000],
        targeted_SVMOPT_jpeg_5000=targeted_result["jpeg"]["SVM-OPT"][5000],
        targeted_SVMOPT_jpeg_8000=targeted_result["jpeg"]["SVM-OPT"][8000],
        targeted_SVMOPT_jpeg_10000=targeted_result["jpeg"]["SVM-OPT"][10000],

        untargeted_SVMOPT_feature_distillation_1000=untargeted_result["feature_distillation"]["SVM-OPT"][1000],
        untargeted_SVMOPT_feature_distillation_2000=untargeted_result["feature_distillation"]["SVM-OPT"][2000],
        untargeted_SVMOPT_feature_distillation_5000=untargeted_result["feature_distillation"]["SVM-OPT"][5000],
        untargeted_SVMOPT_feature_distillation_8000=untargeted_result["feature_distillation"]["SVM-OPT"][8000],
        untargeted_SVMOPT_feature_distillation_10000=untargeted_result["feature_distillation"]["SVM-OPT"][10000],

        targeted_SVMOPT_feature_distillation_1000=targeted_result["feature_distillation"]["SVM-OPT"][1000],
        targeted_SVMOPT_feature_distillation_2000=targeted_result["feature_distillation"]["SVM-OPT"][2000],
        targeted_SVMOPT_feature_distillation_5000=targeted_result["feature_distillation"]["SVM-OPT"][5000],
        targeted_SVMOPT_feature_distillation_8000=targeted_result["feature_distillation"]["SVM-OPT"][8000],
        targeted_SVMOPT_feature_distillation_10000=targeted_result["feature_distillation"]["SVM-OPT"][10000],

        untargeted_SVMOPT_TRADES_1000=untargeted_result["TRADES"]["SVM-OPT"][1000],
        untargeted_SVMOPT_TRADES_2000=untargeted_result["TRADES"]["SVM-OPT"][2000],
        untargeted_SVMOPT_TRADES_5000=untargeted_result["TRADES"]["SVM-OPT"][5000],
        untargeted_SVMOPT_TRADES_8000=untargeted_result["TRADES"]["SVM-OPT"][8000],
        untargeted_SVMOPT_TRADES_10000=untargeted_result["TRADES"]["SVM-OPT"][10000],

        targeted_SVMOPT_TRADES_1000=targeted_result["TRADES"]["SVM-OPT"][1000],
        targeted_SVMOPT_TRADES_2000=targeted_result["TRADES"]["SVM-OPT"][2000],
        targeted_SVMOPT_TRADES_5000=targeted_result["TRADES"]["SVM-OPT"][5000],
        targeted_SVMOPT_TRADES_8000=targeted_result["TRADES"]["SVM-OPT"][8000],
        targeted_SVMOPT_TRADES_10000=targeted_result["TRADES"]["SVM-OPT"][10000],

        untargeted_SVMOPT_feature_scatter_1000=untargeted_result["feature_scatter"]["SVM-OPT"][1000],
        untargeted_SVMOPT_feature_scatter_2000=untargeted_result["feature_scatter"]["SVM-OPT"][2000],
        untargeted_SVMOPT_feature_scatter_5000=untargeted_result["feature_scatter"]["SVM-OPT"][5000],
        untargeted_SVMOPT_feature_scatter_8000=untargeted_result["feature_scatter"]["SVM-OPT"][8000],
        untargeted_SVMOPT_feature_scatter_10000=untargeted_result["feature_scatter"]["SVM-OPT"][10000],

        targeted_SVMOPT_feature_scatter_1000=targeted_result["feature_scatter"]["SVM-OPT"][1000],
        targeted_SVMOPT_feature_scatter_2000=targeted_result["feature_scatter"]["SVM-OPT"][2000],
        targeted_SVMOPT_feature_scatter_5000=targeted_result["feature_scatter"]["SVM-OPT"][5000],
        targeted_SVMOPT_feature_scatter_8000=targeted_result["feature_scatter"]["SVM-OPT"][8000],
        targeted_SVMOPT_feature_scatter_10000=targeted_result["feature_scatter"]["SVM-OPT"][10000],

        untargeted_SVMOPT_com_defend_1000=untargeted_result["com_defend"]["SVM-OPT"][1000],
        untargeted_SVMOPT_com_defend_2000=untargeted_result["com_defend"]["SVM-OPT"][2000],
        untargeted_SVMOPT_com_defend_5000=untargeted_result["com_defend"]["SVM-OPT"][5000],
        untargeted_SVMOPT_com_defend_8000=untargeted_result["com_defend"]["SVM-OPT"][8000],
        untargeted_SVMOPT_com_defend_10000=untargeted_result["com_defend"]["SVM-OPT"][10000],

        targeted_SVMOPT_com_defend_1000=targeted_result["com_defend"]["SVM-OPT"][1000],
        targeted_SVMOPT_com_defend_2000=targeted_result["com_defend"]["SVM-OPT"][2000],
        targeted_SVMOPT_com_defend_5000=targeted_result["com_defend"]["SVM-OPT"][5000],
        targeted_SVMOPT_com_defend_8000=targeted_result["com_defend"]["SVM-OPT"][8000],
        targeted_SVMOPT_com_defend_10000=targeted_result["com_defend"]["SVM-OPT"][10000],

        untargeted_GeoDA_jpeg_1000=untargeted_result["jpeg"]["GeoDA"][1000],
        untargeted_GeoDA_jpeg_2000=untargeted_result["jpeg"]["GeoDA"][2000],
        untargeted_GeoDA_jpeg_5000=untargeted_result["jpeg"]["GeoDA"][5000],
        untargeted_GeoDA_jpeg_8000=untargeted_result["jpeg"]["GeoDA"][8000],
        untargeted_GeoDA_jpeg_10000=untargeted_result["jpeg"]["GeoDA"][10000],

        targeted_GeoDA_jpeg_1000=targeted_result["jpeg"]["GeoDA"][1000],
        targeted_GeoDA_jpeg_2000=targeted_result["jpeg"]["GeoDA"][2000],
        targeted_GeoDA_jpeg_5000=targeted_result["jpeg"]["GeoDA"][5000],
        targeted_GeoDA_jpeg_8000=targeted_result["jpeg"]["GeoDA"][8000],
        targeted_GeoDA_jpeg_10000=targeted_result["jpeg"]["GeoDA"][10000],

        untargeted_GeoDA_feature_distillation_1000=untargeted_result["feature_distillation"]["GeoDA"][1000],
        untargeted_GeoDA_feature_distillation_2000=untargeted_result["feature_distillation"]["GeoDA"][2000],
        untargeted_GeoDA_feature_distillation_5000=untargeted_result["feature_distillation"]["GeoDA"][5000],
        untargeted_GeoDA_feature_distillation_8000=untargeted_result["feature_distillation"]["GeoDA"][8000],
        untargeted_GeoDA_feature_distillation_10000=untargeted_result["feature_distillation"]["GeoDA"][10000],

        targeted_GeoDA_feature_distillation_1000=targeted_result["feature_distillation"]["GeoDA"][1000],
        targeted_GeoDA_feature_distillation_2000=targeted_result["feature_distillation"]["GeoDA"][2000],
        targeted_GeoDA_feature_distillation_5000=targeted_result["feature_distillation"]["GeoDA"][5000],
        targeted_GeoDA_feature_distillation_8000=targeted_result["feature_distillation"]["GeoDA"][8000],
        targeted_GeoDA_feature_distillation_10000=targeted_result["feature_distillation"]["GeoDA"][10000],

        untargeted_GeoDA_TRADES_1000=untargeted_result["TRADES"]["GeoDA"][1000],
        untargeted_GeoDA_TRADES_2000=untargeted_result["TRADES"]["GeoDA"][2000],
        untargeted_GeoDA_TRADES_5000=untargeted_result["TRADES"]["GeoDA"][5000],
        untargeted_GeoDA_TRADES_8000=untargeted_result["TRADES"]["GeoDA"][8000],
        untargeted_GeoDA_TRADES_10000=untargeted_result["TRADES"]["GeoDA"][10000],

        targeted_GeoDA_TRADES_1000=targeted_result["TRADES"]["GeoDA"][1000],
        targeted_GeoDA_TRADES_2000=targeted_result["TRADES"]["GeoDA"][2000],
        targeted_GeoDA_TRADES_5000=targeted_result["TRADES"]["GeoDA"][5000],
        targeted_GeoDA_TRADES_8000=targeted_result["TRADES"]["GeoDA"][8000],
        targeted_GeoDA_TRADES_10000=targeted_result["TRADES"]["GeoDA"][10000],

        untargeted_GeoDA_feature_scatter_1000=untargeted_result["feature_scatter"]["GeoDA"][1000],
        untargeted_GeoDA_feature_scatter_2000=untargeted_result["feature_scatter"]["GeoDA"][2000],
        untargeted_GeoDA_feature_scatter_5000=untargeted_result["feature_scatter"]["GeoDA"][5000],
        untargeted_GeoDA_feature_scatter_8000=untargeted_result["feature_scatter"]["GeoDA"][8000],
        untargeted_GeoDA_feature_scatter_10000=untargeted_result["feature_scatter"]["GeoDA"][10000],

        targeted_GeoDA_feature_scatter_1000=targeted_result["feature_scatter"]["GeoDA"][1000],
        targeted_GeoDA_feature_scatter_2000=targeted_result["feature_scatter"]["GeoDA"][2000],
        targeted_GeoDA_feature_scatter_5000=targeted_result["feature_scatter"]["GeoDA"][5000],
        targeted_GeoDA_feature_scatter_8000=targeted_result["feature_scatter"]["GeoDA"][8000],
        targeted_GeoDA_feature_scatter_10000=targeted_result["feature_scatter"]["GeoDA"][10000],

        untargeted_GeoDA_com_defend_1000=untargeted_result["com_defend"]["GeoDA"][1000],
        untargeted_GeoDA_com_defend_2000=untargeted_result["com_defend"]["GeoDA"][2000],
        untargeted_GeoDA_com_defend_5000=untargeted_result["com_defend"]["GeoDA"][5000],
        untargeted_GeoDA_com_defend_8000=untargeted_result["com_defend"]["GeoDA"][8000],
        untargeted_GeoDA_com_defend_10000=untargeted_result["com_defend"]["GeoDA"][10000],

        targeted_GeoDA_com_defend_1000=targeted_result["com_defend"]["GeoDA"][1000],
        targeted_GeoDA_com_defend_2000=targeted_result["com_defend"]["GeoDA"][2000],
        targeted_GeoDA_com_defend_5000=targeted_result["com_defend"]["GeoDA"][5000],
        targeted_GeoDA_com_defend_8000=targeted_result["com_defend"]["GeoDA"][8000],
        targeted_GeoDA_com_defend_10000=targeted_result["com_defend"]["GeoDA"][10000],

        untargeted_RayS_jpeg_1000=untargeted_result["jpeg"]["RayS"][1000],
        untargeted_RayS_jpeg_2000=untargeted_result["jpeg"]["RayS"][2000],
        untargeted_RayS_jpeg_5000=untargeted_result["jpeg"]["RayS"][5000],
        untargeted_RayS_jpeg_8000=untargeted_result["jpeg"]["RayS"][8000],
        untargeted_RayS_jpeg_10000=untargeted_result["jpeg"]["RayS"][10000],

        targeted_RayS_jpeg_1000=targeted_result["jpeg"]["RayS"][1000],
        targeted_RayS_jpeg_2000=targeted_result["jpeg"]["RayS"][2000],
        targeted_RayS_jpeg_5000=targeted_result["jpeg"]["RayS"][5000],
        targeted_RayS_jpeg_8000=targeted_result["jpeg"]["RayS"][8000],
        targeted_RayS_jpeg_10000=targeted_result["jpeg"]["RayS"][10000],

        untargeted_RayS_feature_distillation_1000=untargeted_result["feature_distillation"]["RayS"][1000],
        untargeted_RayS_feature_distillation_2000=untargeted_result["feature_distillation"]["RayS"][2000],
        untargeted_RayS_feature_distillation_5000=untargeted_result["feature_distillation"]["RayS"][5000],
        untargeted_RayS_feature_distillation_8000=untargeted_result["feature_distillation"]["RayS"][8000],
        untargeted_RayS_feature_distillation_10000=untargeted_result["feature_distillation"]["RayS"][10000],

        targeted_RayS_feature_distillation_1000=targeted_result["feature_distillation"]["RayS"][1000],
        targeted_RayS_feature_distillation_2000=targeted_result["feature_distillation"]["RayS"][2000],
        targeted_RayS_feature_distillation_5000=targeted_result["feature_distillation"]["RayS"][5000],
        targeted_RayS_feature_distillation_8000=targeted_result["feature_distillation"]["RayS"][8000],
        targeted_RayS_feature_distillation_10000=targeted_result["feature_distillation"]["RayS"][10000],

        untargeted_RayS_TRADES_1000=untargeted_result["TRADES"]["RayS"][1000],
        untargeted_RayS_TRADES_2000=untargeted_result["TRADES"]["RayS"][2000],
        untargeted_RayS_TRADES_5000=untargeted_result["TRADES"]["RayS"][5000],
        untargeted_RayS_TRADES_8000=untargeted_result["TRADES"]["RayS"][8000],
        untargeted_RayS_TRADES_10000=untargeted_result["TRADES"]["RayS"][10000],

        targeted_RayS_TRADES_1000=targeted_result["TRADES"]["RayS"][1000],
        targeted_RayS_TRADES_2000=targeted_result["TRADES"]["RayS"][2000],
        targeted_RayS_TRADES_5000=targeted_result["TRADES"]["RayS"][5000],
        targeted_RayS_TRADES_8000=targeted_result["TRADES"]["RayS"][8000],
        targeted_RayS_TRADES_10000=targeted_result["TRADES"]["RayS"][10000],

        untargeted_RayS_feature_scatter_1000=untargeted_result["feature_scatter"]["RayS"][1000],
        untargeted_RayS_feature_scatter_2000=untargeted_result["feature_scatter"]["RayS"][2000],
        untargeted_RayS_feature_scatter_5000=untargeted_result["feature_scatter"]["RayS"][5000],
        untargeted_RayS_feature_scatter_8000=untargeted_result["feature_scatter"]["RayS"][8000],
        untargeted_RayS_feature_scatter_10000=untargeted_result["feature_scatter"]["RayS"][10000],

        targeted_RayS_feature_scatter_1000=targeted_result["feature_scatter"]["RayS"][1000],
        targeted_RayS_feature_scatter_2000=targeted_result["feature_scatter"]["RayS"][2000],
        targeted_RayS_feature_scatter_5000=targeted_result["feature_scatter"]["RayS"][5000],
        targeted_RayS_feature_scatter_8000=targeted_result["feature_scatter"]["RayS"][8000],
        targeted_RayS_feature_scatter_10000=targeted_result["feature_scatter"]["RayS"][10000],

        untargeted_RayS_com_defend_1000=untargeted_result["com_defend"]["RayS"][1000],
        untargeted_RayS_com_defend_2000=untargeted_result["com_defend"]["RayS"][2000],
        untargeted_RayS_com_defend_5000=untargeted_result["com_defend"]["RayS"][5000],
        untargeted_RayS_com_defend_8000=untargeted_result["com_defend"]["RayS"][8000],
        untargeted_RayS_com_defend_10000=untargeted_result["com_defend"]["RayS"][10000],

        targeted_RayS_com_defend_1000=targeted_result["com_defend"]["RayS"][1000],
        targeted_RayS_com_defend_2000=targeted_result["com_defend"]["RayS"][2000],
        targeted_RayS_com_defend_5000=targeted_result["com_defend"]["RayS"][5000],
        targeted_RayS_com_defend_8000=targeted_result["com_defend"]["RayS"][8000],
        targeted_RayS_com_defend_10000=targeted_result["com_defend"]["RayS"][10000],

        untargeted_HSJA_jpeg_1000=untargeted_result["jpeg"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_jpeg_2000=untargeted_result["jpeg"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_jpeg_5000=untargeted_result["jpeg"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_jpeg_8000=untargeted_result["jpeg"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_jpeg_10000=untargeted_result["jpeg"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_jpeg_1000=targeted_result["jpeg"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_jpeg_2000=targeted_result["jpeg"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_jpeg_5000=targeted_result["jpeg"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_jpeg_8000=targeted_result["jpeg"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_jpeg_10000=targeted_result["jpeg"]["HopSkipJumpAttack"][10000],

        untargeted_HSJA_feature_distillation_1000=untargeted_result["feature_distillation"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_feature_distillation_2000=untargeted_result["feature_distillation"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_feature_distillation_5000=untargeted_result["feature_distillation"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_feature_distillation_8000=untargeted_result["feature_distillation"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_feature_distillation_10000=untargeted_result["feature_distillation"]["HopSkipJumpAttack"][
            10000],

        targeted_HSJA_feature_distillation_1000=targeted_result["feature_distillation"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_feature_distillation_2000=targeted_result["feature_distillation"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_feature_distillation_5000=targeted_result["feature_distillation"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_feature_distillation_8000=targeted_result["feature_distillation"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_feature_distillation_10000=targeted_result["feature_distillation"]["HopSkipJumpAttack"][10000],

        untargeted_HSJA_TRADES_1000=untargeted_result["TRADES"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_TRADES_2000=untargeted_result["TRADES"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_TRADES_5000=untargeted_result["TRADES"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_TRADES_8000=untargeted_result["TRADES"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_TRADES_10000=untargeted_result["TRADES"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_TRADES_1000=targeted_result["TRADES"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_TRADES_2000=targeted_result["TRADES"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_TRADES_5000=targeted_result["TRADES"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_TRADES_8000=targeted_result["TRADES"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_TRADES_10000=targeted_result["TRADES"]["HopSkipJumpAttack"][10000],

        untargeted_HSJA_feature_scatter_1000=untargeted_result["feature_scatter"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_feature_scatter_2000=untargeted_result["feature_scatter"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_feature_scatter_5000=untargeted_result["feature_scatter"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_feature_scatter_8000=untargeted_result["feature_scatter"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_feature_scatter_10000=untargeted_result["feature_scatter"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_feature_scatter_1000=targeted_result["feature_scatter"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_feature_scatter_2000=targeted_result["feature_scatter"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_feature_scatter_5000=targeted_result["feature_scatter"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_feature_scatter_8000=targeted_result["feature_scatter"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_feature_scatter_10000=targeted_result["feature_scatter"]["HopSkipJumpAttack"][10000],

        untargeted_HSJA_com_defend_1000=untargeted_result["com_defend"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_com_defend_2000=untargeted_result["com_defend"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_com_defend_5000=untargeted_result["com_defend"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_com_defend_8000=untargeted_result["com_defend"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_com_defend_10000=untargeted_result["com_defend"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_com_defend_1000=targeted_result["com_defend"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_com_defend_2000=targeted_result["com_defend"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_com_defend_5000=targeted_result["com_defend"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_com_defend_8000=targeted_result["com_defend"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_com_defend_10000=targeted_result["com_defend"]["HopSkipJumpAttack"][10000],

        untargeted_Tangent_jpeg_1000=untargeted_result["jpeg"]["Tangent Attack"][1000],
        untargeted_Tangent_jpeg_2000=untargeted_result["jpeg"]["Tangent Attack"][2000],
        untargeted_Tangent_jpeg_5000=untargeted_result["jpeg"]["Tangent Attack"][5000],
        untargeted_Tangent_jpeg_8000=untargeted_result["jpeg"]["Tangent Attack"][8000],
        untargeted_Tangent_jpeg_10000=untargeted_result["jpeg"]["Tangent Attack"][10000],

        targeted_Tangent_jpeg_1000=targeted_result["jpeg"]["Tangent Attack"][1000],
        targeted_Tangent_jpeg_2000=targeted_result["jpeg"]["Tangent Attack"][2000],
        targeted_Tangent_jpeg_5000=targeted_result["jpeg"]["Tangent Attack"][5000],
        targeted_Tangent_jpeg_8000=targeted_result["jpeg"]["Tangent Attack"][8000],
        targeted_Tangent_jpeg_10000=targeted_result["jpeg"]["Tangent Attack"][10000],

        untargeted_Tangent_feature_distillation_1000=untargeted_result["feature_distillation"]["Tangent Attack"][1000],
        untargeted_Tangent_feature_distillation_2000=untargeted_result["feature_distillation"]["Tangent Attack"][2000],
        untargeted_Tangent_feature_distillation_5000=untargeted_result["feature_distillation"]["Tangent Attack"][5000],
        untargeted_Tangent_feature_distillation_8000=untargeted_result["feature_distillation"]["Tangent Attack"][8000],
        untargeted_Tangent_feature_distillation_10000=untargeted_result["feature_distillation"]["Tangent Attack"][
            10000],

        targeted_Tangent_feature_distillation_1000=targeted_result["feature_distillation"]["Tangent Attack"][1000],
        targeted_Tangent_feature_distillation_2000=targeted_result["feature_distillation"]["Tangent Attack"][2000],
        targeted_Tangent_feature_distillation_5000=targeted_result["feature_distillation"]["Tangent Attack"][5000],
        targeted_Tangent_feature_distillation_8000=targeted_result["feature_distillation"]["Tangent Attack"][8000],
        targeted_Tangent_feature_distillation_10000=targeted_result["feature_distillation"]["Tangent Attack"][10000],

        untargeted_Tangent_TRADES_1000=untargeted_result["TRADES"]["Tangent Attack"][1000],
        untargeted_Tangent_TRADES_2000=untargeted_result["TRADES"]["Tangent Attack"][2000],
        untargeted_Tangent_TRADES_5000=untargeted_result["TRADES"]["Tangent Attack"][5000],
        untargeted_Tangent_TRADES_8000=untargeted_result["TRADES"]["Tangent Attack"][8000],
        untargeted_Tangent_TRADES_10000=untargeted_result["TRADES"]["Tangent Attack"][10000],

        targeted_Tangent_TRADES_1000=targeted_result["TRADES"]["Tangent Attack"][1000],
        targeted_Tangent_TRADES_2000=targeted_result["TRADES"]["Tangent Attack"][2000],
        targeted_Tangent_TRADES_5000=targeted_result["TRADES"]["Tangent Attack"][5000],
        targeted_Tangent_TRADES_8000=targeted_result["TRADES"]["Tangent Attack"][8000],
        targeted_Tangent_TRADES_10000=targeted_result["TRADES"]["Tangent Attack"][10000],

        untargeted_Tangent_feature_scatter_1000=untargeted_result["feature_scatter"]["Tangent Attack"][1000],
        untargeted_Tangent_feature_scatter_2000=untargeted_result["feature_scatter"]["Tangent Attack"][2000],
        untargeted_Tangent_feature_scatter_5000=untargeted_result["feature_scatter"]["Tangent Attack"][5000],
        untargeted_Tangent_feature_scatter_8000=untargeted_result["feature_scatter"]["Tangent Attack"][8000],
        untargeted_Tangent_feature_scatter_10000=untargeted_result["feature_scatter"]["Tangent Attack"][10000],

        targeted_Tangent_feature_scatter_1000=targeted_result["feature_scatter"]["Tangent Attack"][1000],
        targeted_Tangent_feature_scatter_2000=targeted_result["feature_scatter"]["Tangent Attack"][2000],
        targeted_Tangent_feature_scatter_5000=targeted_result["feature_scatter"]["Tangent Attack"][5000],
        targeted_Tangent_feature_scatter_8000=targeted_result["feature_scatter"]["Tangent Attack"][8000],
        targeted_Tangent_feature_scatter_10000=targeted_result["feature_scatter"]["Tangent Attack"][10000],

        untargeted_Tangent_com_defend_1000=untargeted_result["com_defend"]["Tangent Attack"][1000],
        untargeted_Tangent_com_defend_2000=untargeted_result["com_defend"]["Tangent Attack"][2000],
        untargeted_Tangent_com_defend_5000=untargeted_result["com_defend"]["Tangent Attack"][5000],
        untargeted_Tangent_com_defend_8000=untargeted_result["com_defend"]["Tangent Attack"][8000],
        untargeted_Tangent_com_defend_10000=untargeted_result["com_defend"]["Tangent Attack"][10000],

        targeted_Tangent_com_defend_1000=targeted_result["com_defend"]["Tangent Attack"][1000],
        targeted_Tangent_com_defend_2000=targeted_result["com_defend"]["Tangent Attack"][2000],
        targeted_Tangent_com_defend_5000=targeted_result["com_defend"]["Tangent Attack"][5000],
        targeted_Tangent_com_defend_8000=targeted_result["com_defend"]["Tangent Attack"][8000],
        targeted_Tangent_com_defend_10000=targeted_result["com_defend"]["Tangent Attack"][10000],
    )
    )


if __name__ == "__main__":
    dataset = "CIFAR-10"
    norm = "l2"
    targeted = True
    if "CIFAR" in dataset:
        archs = ['com_defend',"feature_distillation","feature_scatter", "jpeg", "TRADES"]
    else:
        archs = ["resnext101_64x4d","inceptionv4","senet154","resnet101","inceptionv3"]
    query_budgets = [1000,2000,5000,8000,10000]
    # if targeted:
    #     query_budgets.extend([12000,15000,18000,20000])
    if "CIFAR" in dataset:
        targeted_result = {}
        for arch in archs:
            result = fetch_all_json_content_given_contraint(dataset, norm, True, arch, query_budgets, "mean_distortion")
            targeted_result[arch] = result
        untargeted_result = {}
        for arch in archs:
            result = fetch_all_json_content_given_contraint(dataset, norm, False, arch, query_budgets, "mean_distortion")
            untargeted_result[arch] = result

        draw_wide_table_CIFAR(untargeted_result, targeted_result)
