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
                        "boundary_attack":"Boundary Attack",
                        }
                        # "RayS": "RayS","GeoDA": "GeoDA",
                        #"biased_boundary_attack": "Biased Boundary Attack"}

def from_method_to_dir_path(dataset, method, norm, targeted):
    if method == "tangent_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
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
        if targeted:
            path = "{method}_random_start_point-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                                   target_str="untargeted" if not targeted else "targeted_increment")
        else:
            path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SVMOPT":
        if targeted:
            path = "{method}_random_start_point-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                                   target_str="untargeted" if not targeted else "targeted_increment")
        else:
            path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
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
        for idx, (image_index, query_distortion) in enumerate(distortion_dict.items()):
            if idx==0:
                assert int(image_index) == 0
            query_distortion = {int(float(query)):float(dist) for query,dist in query_distortion.items()}
            queries = np.array(list(query_distortion.keys()))
            queries = np.sort(queries)
            # find_index = bisect.bisect(queries, query_budget) - 1
            # print(len(queries),find_index)
            find_index = np.searchsorted(queries, query_budget, side='right') - 1
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
        file_path = folder + "/{}_result.json".format(arch)
        if method in ["RayS","GeoDA"] and targeted:
            print("{} does not exist!".format(file_path))
            result[method] = defaultdict(lambda : "-")
            continue
        distortion_dict = read_json_and_extract(file_path)
        print(file_path)
        mean_and_median_distortions = get_mean_and_median_distortion_given_query_budgets(distortion_dict, query_budgets,want_key)
        result[method] = mean_and_median_distortions
    return result

def draw_tables_for_ImageNet_targeted(result):
    print("""
                    & \\multirow{{5}}{{*}}{{Inception-v3}} & Sign-OPT & {SignOPT_inceptionv3_1000} & {SignOPT_inceptionv3_2000} & {SignOPT_inceptionv3_5000} & {SignOPT_inceptionv3_8000} & {SignOPT_inceptionv3_10000} & - & - & - & - \\\\
                    & SVM-OPT & {SVMOPT_inceptionv3_1000} & {SVMOPT_inceptionv3_2000} & {SVMOPT_inceptionv3_5000} & {SVMOPT_inceptionv3_8000} & {SVMOPT_inceptionv3_10000} & - & - & - & - \\\\
                    & HopSkipJumpAttack & {HSJA_inceptionv3_1000} & {HSJA_inceptionv3_2000} & {HSJA_inceptionv3_5000} & {HSJA_inceptionv3_8000} & {HSJA_inceptionv3_10000} & - & - & - & - \\\\
                    & Tangent Attack (ours) & {Tangent_inceptionv3_1000} & {Tangent_inceptionv3_2000} & {Tangent_inceptionv3_5000} & {Tangent_inceptionv3_8000} & {Tangent_inceptionv3_10000} & - & - & - & - \\\\
                    \\cmidrule(rl){{2-12}}
                    & \\multirow{{5}}{{*}}{{ResNet-101}} & Sign-OPT & {SignOPT_resnet101_1000} & {SignOPT_resnet101_2000} & {SignOPT_resnet101_5000} & {SignOPT_resnet101_8000} & {SignOPT_resnet101_10000} & - & - & - & - \\\\
                    & SVM-OPT & {SVMOPT_resnet101_1000} & {SVMOPT_resnet101_2000} & {SVMOPT_resnet101_5000} & {SVMOPT_resnet101_8000} & {SVMOPT_resnet101_10000} & - & - & - & - \\\\
                    & HopSkipJumpAttack & {HSJA_resnet101_1000} & {HSJA_resnet101_2000} & {HSJA_resnet101_5000} & {HSJA_resnet101_8000} & {HSJA_resnet101_10000} & - & - & - & - \\\\
                    & Tangent Attack (ours) & {Tangent_resnet101_1000} & {Tangent_resnet101_2000} & {Tangent_resnet101_5000} & {Tangent_resnet101_8000} & {Tangent_resnet101_10000} & - & - & - & - \\\\
                    \\cmidrule(rl){{2-12}}
                    & \\multirow{{5}}{{*}}{{ResNeXt-101}} & Sign-OPT & {SignOPT_resnext101_1000} & {SignOPT_resnext101_2000} & {SignOPT_resnext101_5000} & {SignOPT_resnext101_8000} & {SignOPT_resnext101_10000} & - & - & - & - \\\\
                    & SVM-OPT & {SVMOPT_resnext101_1000} & {SVMOPT_resnext101_2000} & {SVMOPT_resnext101_5000} & {SVMOPT_resnext101_8000} & {SVMOPT_resnext101_10000} & - & - & - & - \\\\
                    & HopSkipJumpAttack & {HSJA_resnext101_1000} & {HSJA_resnext101_2000} & {HSJA_resnext101_5000} & {HSJA_resnext101_8000} & {HSJA_resnext101_10000} & - & - & - & - \\\\
                    &  Tangent Attack (ours) & {Tangent_resnext101_1000} & {Tangent_resnext101_2000} & {Tangent_resnext101_5000} & {Tangent_resnext101_8000} & {Tangent_resnext101_10000} & - & - & - & - \\\\
                        """.format(

        SignOPT_inceptionv3_1000=result["inceptionv3"]["Sign-OPT"][1000],
        SignOPT_inceptionv3_2000=result["inceptionv3"]["Sign-OPT"][2000],
        SignOPT_inceptionv3_5000=result["inceptionv3"]["Sign-OPT"][5000],
        SignOPT_inceptionv3_8000=result["inceptionv3"]["Sign-OPT"][8000],
        SignOPT_inceptionv3_10000=result["inceptionv3"]["Sign-OPT"][10000],

        SignOPT_resnet101_1000=result["resnet101"]["Sign-OPT"][1000],
        SignOPT_resnet101_2000=result["resnet101"]["Sign-OPT"][2000],
        SignOPT_resnet101_5000=result["resnet101"]["Sign-OPT"][5000],
        SignOPT_resnet101_8000=result["resnet101"]["Sign-OPT"][8000],
        SignOPT_resnet101_10000=result["resnet101"]["Sign-OPT"][10000],

        SignOPT_resnext101_1000=result["resnext101_64x4d"]["Sign-OPT"][1000],
        SignOPT_resnext101_2000=result["resnext101_64x4d"]["Sign-OPT"][2000],
        SignOPT_resnext101_5000=result["resnext101_64x4d"]["Sign-OPT"][5000],
        SignOPT_resnext101_8000=result["resnext101_64x4d"]["Sign-OPT"][8000],
        SignOPT_resnext101_10000=result["resnext101_64x4d"]["Sign-OPT"][10000],

        SVMOPT_inceptionv3_1000=result["inceptionv3"]["SVM-OPT"][1000],
        SVMOPT_inceptionv3_2000=result["inceptionv3"]["SVM-OPT"][2000],
        SVMOPT_inceptionv3_5000=result["inceptionv3"]["SVM-OPT"][5000],
        SVMOPT_inceptionv3_8000=result["inceptionv3"]["SVM-OPT"][8000],
        SVMOPT_inceptionv3_10000=result["inceptionv3"]["SVM-OPT"][10000],

        SVMOPT_resnet101_1000=result["resnet101"]["SVM-OPT"][1000],
        SVMOPT_resnet101_2000=result["resnet101"]["SVM-OPT"][2000],
        SVMOPT_resnet101_5000=result["resnet101"]["SVM-OPT"][5000],
        SVMOPT_resnet101_8000=result["resnet101"]["SVM-OPT"][8000],
        SVMOPT_resnet101_10000=result["resnet101"]["SVM-OPT"][10000],

        SVMOPT_resnext101_1000=result["resnext101_64x4d"]["SVM-OPT"][1000],
        SVMOPT_resnext101_2000=result["resnext101_64x4d"]["SVM-OPT"][2000],
        SVMOPT_resnext101_5000=result["resnext101_64x4d"]["SVM-OPT"][5000],
        SVMOPT_resnext101_8000=result["resnext101_64x4d"]["SVM-OPT"][8000],
        SVMOPT_resnext101_10000=result["resnext101_64x4d"]["SVM-OPT"][10000],

        HSJA_inceptionv3_1000=result["inceptionv3"]["HopSkipJumpAttack"][1000],
        HSJA_inceptionv3_2000=result["inceptionv3"]["HopSkipJumpAttack"][2000],
        HSJA_inceptionv3_5000=result["inceptionv3"]["HopSkipJumpAttack"][5000],
        HSJA_inceptionv3_8000=result["inceptionv3"]["HopSkipJumpAttack"][8000],
        HSJA_inceptionv3_10000=result["inceptionv3"]["HopSkipJumpAttack"][10000],

        HSJA_resnet101_1000=result["resnet101"]["HopSkipJumpAttack"][1000],
        HSJA_resnet101_2000=result["resnet101"]["HopSkipJumpAttack"][2000],
        HSJA_resnet101_5000=result["resnet101"]["HopSkipJumpAttack"][5000],
        HSJA_resnet101_8000=result["resnet101"]["HopSkipJumpAttack"][8000],
        HSJA_resnet101_10000=result["resnet101"]["HopSkipJumpAttack"][10000],

        HSJA_resnext101_1000=result["resnext101_64x4d"]["HopSkipJumpAttack"][1000],
        HSJA_resnext101_2000=result["resnext101_64x4d"]["HopSkipJumpAttack"][2000],
        HSJA_resnext101_5000=result["resnext101_64x4d"]["HopSkipJumpAttack"][5000],
        HSJA_resnext101_8000=result["resnext101_64x4d"]["HopSkipJumpAttack"][8000],
        HSJA_resnext101_10000=result["resnext101_64x4d"]["HopSkipJumpAttack"][10000],

        Tangent_inceptionv3_1000=result["inceptionv3"]["Tangent Attack"][1000],
        Tangent_inceptionv3_2000=result["inceptionv3"]["Tangent Attack"][2000],
        Tangent_inceptionv3_5000=result["inceptionv3"]["Tangent Attack"][5000],
        Tangent_inceptionv3_8000=result["inceptionv3"]["Tangent Attack"][8000],
        Tangent_inceptionv3_10000=result["inceptionv3"]["Tangent Attack"][10000],

        Tangent_resnet101_1000=result["resnet101"]["Tangent Attack"][1000],
        Tangent_resnet101_2000=result["resnet101"]["Tangent Attack"][2000],
        Tangent_resnet101_5000=result["resnet101"]["Tangent Attack"][5000],
        Tangent_resnet101_8000=result["resnet101"]["Tangent Attack"][8000],
        Tangent_resnet101_10000=result["resnet101"]["Tangent Attack"][10000],

        Tangent_resnext101_1000=result["resnext101_64x4d"]["Tangent Attack"][1000],
        Tangent_resnext101_2000=result["resnext101_64x4d"]["Tangent Attack"][2000],
        Tangent_resnext101_5000=result["resnext101_64x4d"]["Tangent Attack"][5000],
        Tangent_resnext101_8000=result["resnext101_64x4d"]["Tangent Attack"][8000],
        Tangent_resnext101_10000=result["resnext101_64x4d"]["Tangent Attack"][10000],
    ))


def draw_tables_for_ImageNet(result):
    print("""
                    & Boundary Attack & {BoundaryAttack_inceptionv3_1000} & {BoundaryAttack_inceptionv3_2000} & {BoundaryAttack_inceptionv3_5000} & {BoundaryAttack_inceptionv3_8000} & {BoundaryAttack_inceptionv3_10000} \\\\
                    & Sign-OPT & {SignOPT_inceptionv3_1000} & {SignOPT_inceptionv3_2000} & {SignOPT_inceptionv3_5000} & {SignOPT_inceptionv3_8000} & {SignOPT_inceptionv3_10000} \\\\
                    & SVM-OPT & {SVMOPT_inceptionv3_1000} & {SVMOPT_inceptionv3_2000} & {SVMOPT_inceptionv3_5000} & {SVMOPT_inceptionv3_8000} & {SVMOPT_inceptionv3_10000} \\\\
                    & HopSkipJumpAttack & {HSJA_inceptionv3_1000} & {HSJA_inceptionv3_2000} & {HSJA_inceptionv3_5000} & {HSJA_inceptionv3_8000} & {HSJA_inceptionv3_10000} \\\\
                    & Tangent Attack (ours)  {Tangent_inceptionv3_1000} & {Tangent_inceptionv3_2000} & {Tangent_inceptionv3_5000} & {Tangent_inceptionv3_8000} & {Tangent_inceptionv3_10000} \\\\
                    \\midrule
                    & Boundary Attack & {BoundaryAttack_inceptionv4_1000} & {BoundaryAttack_inceptionv4_2000} & {BoundaryAttack_inceptionv4_5000} & {BoundaryAttack_inceptionv4_8000} & {BoundaryAttack_inceptionv4_10000} \\\\
                    & Sign-OPT& {SignOPT_inceptionv4_1000} & {SignOPT_inceptionv4_2000} & {SignOPT_inceptionv4_5000} & {SignOPT_inceptionv4_8000} & {SignOPT_inceptionv4_10000} \\\\
                    & SVM-OPT & {SVMOPT_inceptionv4_1000} & {SVMOPT_inceptionv4_2000} & {SVMOPT_inceptionv4_5000} & {SVMOPT_inceptionv4_8000} & {SVMOPT_inceptionv4_10000} \\\\
                    & HopSkipJumpAttack & {HSJA_inceptionv4_1000} & {HSJA_inceptionv4_2000} & {HSJA_inceptionv4_5000} & {HSJA_inceptionv4_8000} & {HSJA_inceptionv4_10000} \\\\
                    & Tangent Attack (ours) & {Tangent_inceptionv4_1000} & {Tangent_inceptionv4_2000} & {Tangent_inceptionv4_5000} & {Tangent_inceptionv4_8000} & {Tangent_inceptionv4_10000} \\\\
                    \\midrule
                    & Boundary Attack  & {BoundaryAttack_senet154_1000} & {BoundaryAttack_senet154_2000} & {BoundaryAttack_senet154_5000} & {BoundaryAttack_senet154_8000} & {BoundaryAttack_senet154_10000} \\\\
                    & Sign-OPT & {SignOPT_senet154_1000} & {SignOPT_senet154_2000} & {SignOPT_senet154_5000} & {SignOPT_senet154_8000} & {SignOPT_senet154_10000} \\\\
                    & SVM-OPT & {SVMOPT_senet154_1000} & {SVMOPT_senet154_2000} & {SVMOPT_senet154_5000} & {SVMOPT_senet154_8000} & {SVMOPT_senet154_10000} \\\\
                    & HopSkipJumpAttack & {HSJA_senet154_1000} & {HSJA_senet154_2000} & {HSJA_senet154_5000} & {HSJA_senet154_8000} & {HSJA_senet154_10000} \\\\
                    & Tangent Attack (ours) & {Tangent_senet154_1000} & {Tangent_senet154_2000} & {Tangent_senet154_5000} & {Tangent_senet154_8000} & {Tangent_senet154_10000} \\\\
                    \\midrule
                    & Boundary Attack & {BoundaryAttack_resnet101_1000} & {BoundaryAttack_resnet101_2000} & {BoundaryAttack_resnet101_5000} & {BoundaryAttack_resnet101_8000} & {BoundaryAttack_resnet101_10000} \\\\
                    & Sign-OPT & {SignOPT_resnet101_1000} & {SignOPT_resnet101_2000} & {SignOPT_resnet101_5000} & {SignOPT_resnet101_8000} & {SignOPT_resnet101_10000} \\\\
                    & SVM-OPT  & {SVMOPT_resnet101_1000} & {SVMOPT_resnet101_2000} & {SVMOPT_resnet101_5000} & {SVMOPT_resnet101_8000} & {SVMOPT_resnet101_10000} \\\\                    
                    & HopSkipJumpAttack & {HSJA_resnet101_1000} & {HSJA_resnet101_2000} & {HSJA_resnet101_5000} & {HSJA_resnet101_8000} & {HSJA_resnet101_10000} \\\\
                    & Tangent Attack (ours) & {Tangent_resnet101_1000} & {Tangent_resnet101_2000} & {Tangent_resnet101_5000} & {Tangent_resnet101_8000} & {Tangent_resnet101_10000} \\\\
                    \\cmidrule(rl){{2-12}}
                    & Boundary Attack  & {BoundaryAttack_resnext101_1000} & {BoundaryAttack_resnext101_2000} & {BoundaryAttack_resnext101_5000} & {BoundaryAttack_resnext101_8000} & {BoundaryAttack_resnext101_10000} \\\\
                    & Sign-OPT & {SignOPT_resnext101_1000} & {SignOPT_resnext101_2000} & {SignOPT_resnext101_5000} & {SignOPT_resnext101_8000} & {SignOPT_resnext101_10000} \\\\
                    & SVM-OPT  & {SVMOPT_resnext101_1000} & {SVMOPT_resnext101_2000} & {SVMOPT_resnext101_5000} & {SVMOPT_resnext101_8000} & {SVMOPT_resnext101_10000} \\\\
                    & HopSkipJumpAttack & {HSJA_resnext101_1000} & {HSJA_resnext101_2000} & {HSJA_resnext101_5000} & {HSJA_resnext101_8000} & {HSJA_resnext101_10000} \\\\
                    & Tangent Attack (ours) & {Tangent_resnext101_1000} & {Tangent_resnext101_2000} & {Tangent_resnext101_5000} & {Tangent_resnext101_8000} & {Tangent_resnext101_10000} \\\\
                        """.format(

        SignOPT_inceptionv3_1000=result["inceptionv3"]["Sign-OPT"][1000],
        SignOPT_inceptionv3_2000=result["inceptionv3"]["Sign-OPT"][2000],
        SignOPT_inceptionv3_5000=result["inceptionv3"]["Sign-OPT"][5000],
        SignOPT_inceptionv3_8000=result["inceptionv3"]["Sign-OPT"][8000],
        SignOPT_inceptionv3_10000=result["inceptionv3"]["Sign-OPT"][10000],

        SignOPT_inceptionv4_1000=result["inceptionv4"]["Sign-OPT"][1000],
        SignOPT_inceptionv4_2000=result["inceptionv4"]["Sign-OPT"][2000],
        SignOPT_inceptionv4_5000=result["inceptionv4"]["Sign-OPT"][5000],
        SignOPT_inceptionv4_8000=result["inceptionv4"]["Sign-OPT"][8000],
        SignOPT_inceptionv4_10000=result["inceptionv4"]["Sign-OPT"][10000],

        SignOPT_senet154_1000=result["senet154"]["Sign-OPT"][1000],
        SignOPT_senet154_2000=result["senet154"]["Sign-OPT"][2000],
        SignOPT_senet154_5000=result["senet154"]["Sign-OPT"][5000],
        SignOPT_senet154_8000=result["senet154"]["Sign-OPT"][8000],
        SignOPT_senet154_10000=result["senet154"]["Sign-OPT"][10000],

        SignOPT_resnet101_1000=result["resnet101"]["Sign-OPT"][1000],
        SignOPT_resnet101_2000=result["resnet101"]["Sign-OPT"][2000],
        SignOPT_resnet101_5000=result["resnet101"]["Sign-OPT"][5000],
        SignOPT_resnet101_8000=result["resnet101"]["Sign-OPT"][8000],
        SignOPT_resnet101_10000=result["resnet101"]["Sign-OPT"][10000],

        SignOPT_resnext101_1000=result["resnext101_64x4d"]["Sign-OPT"][1000],
        SignOPT_resnext101_2000=result["resnext101_64x4d"]["Sign-OPT"][2000],
        SignOPT_resnext101_5000=result["resnext101_64x4d"]["Sign-OPT"][5000],
        SignOPT_resnext101_8000=result["resnext101_64x4d"]["Sign-OPT"][8000],
        SignOPT_resnext101_10000=result["resnext101_64x4d"]["Sign-OPT"][10000],

        SVMOPT_inceptionv3_1000=result["inceptionv3"]["SVM-OPT"][1000],
        SVMOPT_inceptionv3_2000=result["inceptionv3"]["SVM-OPT"][2000],
        SVMOPT_inceptionv3_5000=result["inceptionv3"]["SVM-OPT"][5000],
        SVMOPT_inceptionv3_8000=result["inceptionv3"]["SVM-OPT"][8000],
        SVMOPT_inceptionv3_10000=result["inceptionv3"]["SVM-OPT"][10000],

        SVMOPT_inceptionv4_1000=result["inceptionv4"]["SVM-OPT"][1000],
        SVMOPT_inceptionv4_2000=result["inceptionv4"]["SVM-OPT"][2000],
        SVMOPT_inceptionv4_5000=result["inceptionv4"]["SVM-OPT"][5000],
        SVMOPT_inceptionv4_8000=result["inceptionv4"]["SVM-OPT"][8000],
        SVMOPT_inceptionv4_10000=result["inceptionv4"]["SVM-OPT"][10000],

        SVMOPT_senet154_1000=result["senet154"]["SVM-OPT"][1000],
        SVMOPT_senet154_2000=result["senet154"]["SVM-OPT"][2000],
        SVMOPT_senet154_5000=result["senet154"]["SVM-OPT"][5000],
        SVMOPT_senet154_8000=result["senet154"]["SVM-OPT"][8000],
        SVMOPT_senet154_10000=result["senet154"]["SVM-OPT"][10000],

        SVMOPT_resnet101_1000=result["resnet101"]["SVM-OPT"][1000],
        SVMOPT_resnet101_2000=result["resnet101"]["SVM-OPT"][2000],
        SVMOPT_resnet101_5000=result["resnet101"]["SVM-OPT"][5000],
        SVMOPT_resnet101_8000=result["resnet101"]["SVM-OPT"][8000],
        SVMOPT_resnet101_10000=result["resnet101"]["SVM-OPT"][10000],

        SVMOPT_resnext101_1000=result["resnext101_64x4d"]["SVM-OPT"][1000],
        SVMOPT_resnext101_2000=result["resnext101_64x4d"]["SVM-OPT"][2000],
        SVMOPT_resnext101_5000=result["resnext101_64x4d"]["SVM-OPT"][5000],
        SVMOPT_resnext101_8000=result["resnext101_64x4d"]["SVM-OPT"][8000],
        SVMOPT_resnext101_10000=result["resnext101_64x4d"]["SVM-OPT"][10000],

        BoundaryAttack_inceptionv3_1000=result["inceptionv3"]["Boundary Attack"][1000],
        BoundaryAttack_inceptionv3_2000=result["inceptionv3"]["Boundary Attack"][2000],
        BoundaryAttack_inceptionv3_5000=result["inceptionv3"]["Boundary Attack"][5000],
        BoundaryAttack_inceptionv3_8000=result["inceptionv3"]["Boundary Attack"][8000],
        BoundaryAttack_inceptionv3_10000=result["inceptionv3"]["Boundary Attack"][10000],

        BoundaryAttack_inceptionv4_1000=result["inceptionv4"]["Boundary Attack"][1000],
        BoundaryAttack_inceptionv4_2000=result["inceptionv4"]["Boundary Attack"][2000],
        BoundaryAttack_inceptionv4_5000=result["inceptionv4"]["Boundary Attack"][5000],
        BoundaryAttack_inceptionv4_8000=result["inceptionv4"]["Boundary Attack"][8000],
        BoundaryAttack_inceptionv4_10000=result["inceptionv4"]["Boundary Attack"][10000],

        BoundaryAttack_senet154_1000=result["senet154"]["Boundary Attack"][1000],
        BoundaryAttack_senet154_2000=result["senet154"]["Boundary Attack"][2000],
        BoundaryAttack_senet154_5000=result["senet154"]["Boundary Attack"][5000],
        BoundaryAttack_senet154_8000=result["senet154"]["Boundary Attack"][8000],
        BoundaryAttack_senet154_10000=result["senet154"]["Boundary Attack"][10000],

        BoundaryAttack_resnet101_1000=result["resnet101"]["Boundary Attack"][1000],
        BoundaryAttack_resnet101_2000=result["resnet101"]["Boundary Attack"][2000],
        BoundaryAttack_resnet101_5000=result["resnet101"]["Boundary Attack"][5000],
        BoundaryAttack_resnet101_8000=result["resnet101"]["Boundary Attack"][8000],
        BoundaryAttack_resnet101_10000=result["resnet101"]["Boundary Attack"][10000],

        BoundaryAttack_resnext101_1000=result["resnext101_64x4d"]["Boundary Attack"][1000],
        BoundaryAttack_resnext101_2000=result["resnext101_64x4d"]["Boundary Attack"][2000],
        BoundaryAttack_resnext101_5000=result["resnext101_64x4d"]["Boundary Attack"][5000],
        BoundaryAttack_resnext101_8000=result["resnext101_64x4d"]["Boundary Attack"][8000],
        BoundaryAttack_resnext101_10000=result["resnext101_64x4d"]["Boundary Attack"][10000],

        HSJA_inceptionv3_1000=result["inceptionv3"]["HopSkipJumpAttack"][1000],
        HSJA_inceptionv3_2000=result["inceptionv3"]["HopSkipJumpAttack"][2000],
        HSJA_inceptionv3_5000=result["inceptionv3"]["HopSkipJumpAttack"][5000],
        HSJA_inceptionv3_8000=result["inceptionv3"]["HopSkipJumpAttack"][8000],
        HSJA_inceptionv3_10000=result["inceptionv3"]["HopSkipJumpAttack"][10000],

        HSJA_inceptionv4_1000=result["inceptionv4"]["HopSkipJumpAttack"][1000],
        HSJA_inceptionv4_2000=result["inceptionv4"]["HopSkipJumpAttack"][2000],
        HSJA_inceptionv4_5000=result["inceptionv4"]["HopSkipJumpAttack"][5000],
        HSJA_inceptionv4_8000=result["inceptionv4"]["HopSkipJumpAttack"][8000],
        HSJA_inceptionv4_10000=result["inceptionv4"]["HopSkipJumpAttack"][10000],

        HSJA_senet154_1000=result["senet154"]["HopSkipJumpAttack"][1000],
        HSJA_senet154_2000=result["senet154"]["HopSkipJumpAttack"][2000],
        HSJA_senet154_5000=result["senet154"]["HopSkipJumpAttack"][5000],
        HSJA_senet154_8000=result["senet154"]["HopSkipJumpAttack"][8000],
        HSJA_senet154_10000=result["senet154"]["HopSkipJumpAttack"][10000],

        HSJA_resnet101_1000=result["resnet101"]["HopSkipJumpAttack"][1000],
        HSJA_resnet101_2000=result["resnet101"]["HopSkipJumpAttack"][2000],
        HSJA_resnet101_5000=result["resnet101"]["HopSkipJumpAttack"][5000],
        HSJA_resnet101_8000=result["resnet101"]["HopSkipJumpAttack"][8000],
        HSJA_resnet101_10000=result["resnet101"]["HopSkipJumpAttack"][10000],

        HSJA_resnext101_1000=result["resnext101_64x4d"]["HopSkipJumpAttack"][1000],
        HSJA_resnext101_2000=result["resnext101_64x4d"]["HopSkipJumpAttack"][2000],
        HSJA_resnext101_5000=result["resnext101_64x4d"]["HopSkipJumpAttack"][5000],
        HSJA_resnext101_8000=result["resnext101_64x4d"]["HopSkipJumpAttack"][8000],
        HSJA_resnext101_10000=result["resnext101_64x4d"]["HopSkipJumpAttack"][10000],

        Tangent_inceptionv3_1000=result["inceptionv3"]["Tangent Attack"][1000],
        Tangent_inceptionv3_2000=result["inceptionv3"]["Tangent Attack"][2000],
        Tangent_inceptionv3_5000=result["inceptionv3"]["Tangent Attack"][5000],
        Tangent_inceptionv3_8000=result["inceptionv3"]["Tangent Attack"][8000],
        Tangent_inceptionv3_10000=result["inceptionv3"]["Tangent Attack"][10000],

        Tangent_inceptionv4_1000=result["inceptionv4"]["Tangent Attack"][1000],
        Tangent_inceptionv4_2000=result["inceptionv4"]["Tangent Attack"][2000],
        Tangent_inceptionv4_5000=result["inceptionv4"]["Tangent Attack"][5000],
        Tangent_inceptionv4_8000=result["inceptionv4"]["Tangent Attack"][8000],
        Tangent_inceptionv4_10000=result["inceptionv4"]["Tangent Attack"][10000],

        Tangent_senet154_1000=result["senet154"]["Tangent Attack"][1000],
        Tangent_senet154_2000=result["senet154"]["Tangent Attack"][2000],
        Tangent_senet154_5000=result["senet154"]["Tangent Attack"][5000],
        Tangent_senet154_8000=result["senet154"]["Tangent Attack"][8000],
        Tangent_senet154_10000=result["senet154"]["Tangent Attack"][10000],

        Tangent_resnet101_1000=result["resnet101"]["Tangent Attack"][1000],
        Tangent_resnet101_2000=result["resnet101"]["Tangent Attack"][2000],
        Tangent_resnet101_5000=result["resnet101"]["Tangent Attack"][5000],
        Tangent_resnet101_8000=result["resnet101"]["Tangent Attack"][8000],
        Tangent_resnet101_10000=result["resnet101"]["Tangent Attack"][10000],

        Tangent_resnext101_1000=result["resnext101_64x4d"]["Tangent Attack"][1000],
        Tangent_resnext101_2000=result["resnext101_64x4d"]["Tangent Attack"][2000],
        Tangent_resnext101_5000=result["resnext101_64x4d"]["Tangent Attack"][5000],
        Tangent_resnext101_8000=result["resnext101_64x4d"]["Tangent Attack"][8000],
        Tangent_resnext101_10000=result["resnext101_64x4d"]["Tangent Attack"][10000],
    )
    )

def draw_tables_for_ImageNet_with_300_query(result):
    print("""
                    & Boundary Attack & {BoundaryAttack_inceptionv3_300} & {BoundaryAttack_inceptionv3_1000} & {BoundaryAttack_inceptionv3_2000} & {BoundaryAttack_inceptionv3_5000} & {BoundaryAttack_inceptionv3_8000} & {BoundaryAttack_inceptionv3_10000} \\\\
                    & Sign-OPT & {SignOPT_inceptionv3_300} & {SignOPT_inceptionv3_1000} & {SignOPT_inceptionv3_2000} & {SignOPT_inceptionv3_5000} & {SignOPT_inceptionv3_8000} & {SignOPT_inceptionv3_10000} \\\\
                    & SVM-OPT & {SVMOPT_inceptionv3_300} & {SVMOPT_inceptionv3_1000} & {SVMOPT_inceptionv3_2000} & {SVMOPT_inceptionv3_5000} & {SVMOPT_inceptionv3_8000} & {SVMOPT_inceptionv3_10000} \\\\
                    & HopSkipJumpAttack & {HSJA_inceptionv3_300} & {HSJA_inceptionv3_1000} & {HSJA_inceptionv3_2000} & {HSJA_inceptionv3_5000} & {HSJA_inceptionv3_8000} & {HSJA_inceptionv3_10000} \\\\
                    & Tangent Attack (ours) & {Tangent_inceptionv3_300} & {Tangent_inceptionv3_1000} & {Tangent_inceptionv3_2000} & {Tangent_inceptionv3_5000} & {Tangent_inceptionv3_8000} & {Tangent_inceptionv3_10000} \\\\
                    \\midrule
                    & Boundary Attack & {BoundaryAttack_inceptionv4_300} & {BoundaryAttack_inceptionv4_1000} & {BoundaryAttack_inceptionv4_2000} & {BoundaryAttack_inceptionv4_5000} & {BoundaryAttack_inceptionv4_8000} & {BoundaryAttack_inceptionv4_10000} \\\\
                    & Sign-OPT & {SignOPT_inceptionv4_300} & {SignOPT_inceptionv4_1000} & {SignOPT_inceptionv4_2000} & {SignOPT_inceptionv4_5000} & {SignOPT_inceptionv4_8000} & {SignOPT_inceptionv4_10000} \\\\
                    & SVM-OPT & {SVMOPT_inceptionv4_300} & {SVMOPT_inceptionv4_1000} & {SVMOPT_inceptionv4_2000} & {SVMOPT_inceptionv4_5000} & {SVMOPT_inceptionv4_8000} & {SVMOPT_inceptionv4_10000} \\\\
                    & HopSkipJumpAttack & {HSJA_inceptionv4_300} & {HSJA_inceptionv4_1000} & {HSJA_inceptionv4_2000} & {HSJA_inceptionv4_5000} & {HSJA_inceptionv4_8000} & {HSJA_inceptionv4_10000} \\\\
                    & Tangent Attack (ours) & {Tangent_inceptionv4_300} & {Tangent_inceptionv4_1000} & {Tangent_inceptionv4_2000} & {Tangent_inceptionv4_5000} & {Tangent_inceptionv4_8000} & {Tangent_inceptionv4_10000} \\\\
                    \\midrule
                    & Boundary Attack & {BoundaryAttack_senet154_300} & {BoundaryAttack_senet154_1000} & {BoundaryAttack_senet154_2000} & {BoundaryAttack_senet154_5000} & {BoundaryAttack_senet154_8000} & {BoundaryAttack_senet154_10000} \\\\
                    & Sign-OPT & {SignOPT_senet154_300} & {SignOPT_senet154_1000} & {SignOPT_senet154_2000} & {SignOPT_senet154_5000} & {SignOPT_senet154_8000} & {SignOPT_senet154_10000} \\\\
                    & SVM-OPT & {SVMOPT_senet154_300} & {SVMOPT_senet154_1000} & {SVMOPT_senet154_2000} & {SVMOPT_senet154_5000} & {SVMOPT_senet154_8000} & {SVMOPT_senet154_10000} \\\\
                    & HopSkipJumpAttack & {HSJA_senet154_300} & {HSJA_senet154_1000} & {HSJA_senet154_2000} & {HSJA_senet154_5000} & {HSJA_senet154_8000} & {HSJA_senet154_10000} \\\\
                    & Tangent Attack (ours) & {Tangent_senet154_300} & {Tangent_senet154_1000} & {Tangent_senet154_2000} & {Tangent_senet154_5000} & {Tangent_senet154_8000} & {Tangent_senet154_10000} \\\\
                    \\midrule
                    & Boundary Attack & {BoundaryAttack_resnet101_300} & {BoundaryAttack_resnet101_1000} & {BoundaryAttack_resnet101_2000} & {BoundaryAttack_resnet101_5000} & {BoundaryAttack_resnet101_8000} & {BoundaryAttack_resnet101_10000} \\\\
                   & Sign-OPT & {SignOPT_resnet101_300} & {SignOPT_resnet101_1000} & {SignOPT_resnet101_2000} & {SignOPT_resnet101_5000} & {SignOPT_resnet101_8000} & {SignOPT_resnet101_10000} \\\\
                    & SVM-OPT & {SVMOPT_resnet101_300} & {SVMOPT_resnet101_1000} & {SVMOPT_resnet101_2000} & {SVMOPT_resnet101_5000} & {SVMOPT_resnet101_8000} & {SVMOPT_resnet101_10000} \\\\
                    & HopSkipJumpAttack & {HSJA_resnet101_300} & {HSJA_resnet101_1000} & {HSJA_resnet101_2000} & {HSJA_resnet101_5000} & {HSJA_resnet101_8000} & {HSJA_resnet101_10000} \\\\
                    & Tangent Attack (ours) & {Tangent_resnet101_300} & {Tangent_resnet101_1000} & {Tangent_resnet101_2000} & {Tangent_resnet101_5000} & {Tangent_resnet101_8000} & {Tangent_resnet101_10000} \\\\
                    \\cmidrule(rl){{2-12}}
                    & Boundary Attack & {BoundaryAttack_resnext101_300} & {BoundaryAttack_resnext101_1000} & {BoundaryAttack_resnext101_2000} & {BoundaryAttack_resnext101_5000} & {BoundaryAttack_resnext101_8000} & {BoundaryAttack_resnext101_10000} \\\\
                    & Sign-OPT & {SignOPT_resnext101_300} & {SignOPT_resnext101_1000} & {SignOPT_resnext101_2000} & {SignOPT_resnext101_5000} & {SignOPT_resnext101_8000} & {SignOPT_resnext101_10000} \\\\
                    & SVM-OPT & {SVMOPT_resnext101_300} & {SVMOPT_resnext101_1000} & {SVMOPT_resnext101_2000} & {SVMOPT_resnext101_5000} & {SVMOPT_resnext101_8000} & {SVMOPT_resnext101_10000} \\\\
                    & HopSkipJumpAttack & {HSJA_resnext101_300} & {HSJA_resnext101_1000} & {HSJA_resnext101_2000} & {HSJA_resnext101_5000} & {HSJA_resnext101_8000} & {HSJA_resnext101_10000} \\\\
                    & Tangent Attack (ours) & {Tangent_resnext101_300} & {Tangent_resnext101_1000} & {Tangent_resnext101_2000} & {Tangent_resnext101_5000} & {Tangent_resnext101_8000} & {Tangent_resnext101_10000} \\\\
                        """.format(

        SignOPT_inceptionv3_300=result["inceptionv3"]["Sign-OPT"][300],
        SignOPT_inceptionv3_1000=result["inceptionv3"]["Sign-OPT"][1000],
        SignOPT_inceptionv3_2000=result["inceptionv3"]["Sign-OPT"][2000],
        SignOPT_inceptionv3_5000=result["inceptionv3"]["Sign-OPT"][5000],
        SignOPT_inceptionv3_8000=result["inceptionv3"]["Sign-OPT"][8000],
        SignOPT_inceptionv3_10000=result["inceptionv3"]["Sign-OPT"][10000],

        SignOPT_inceptionv4_300=result["inceptionv4"]["Sign-OPT"][300],
        SignOPT_inceptionv4_1000=result["inceptionv4"]["Sign-OPT"][1000],
        SignOPT_inceptionv4_2000=result["inceptionv4"]["Sign-OPT"][2000],
        SignOPT_inceptionv4_5000=result["inceptionv4"]["Sign-OPT"][5000],
        SignOPT_inceptionv4_8000=result["inceptionv4"]["Sign-OPT"][8000],
        SignOPT_inceptionv4_10000=result["inceptionv4"]["Sign-OPT"][10000],

        SignOPT_senet154_300=result["senet154"]["Sign-OPT"][300],
        SignOPT_senet154_1000=result["senet154"]["Sign-OPT"][1000],
        SignOPT_senet154_2000=result["senet154"]["Sign-OPT"][2000],
        SignOPT_senet154_5000=result["senet154"]["Sign-OPT"][5000],
        SignOPT_senet154_8000=result["senet154"]["Sign-OPT"][8000],
        SignOPT_senet154_10000=result["senet154"]["Sign-OPT"][10000],

        SignOPT_resnet101_300=result["resnet101"]["Sign-OPT"][300],
        SignOPT_resnet101_1000=result["resnet101"]["Sign-OPT"][1000],
        SignOPT_resnet101_2000=result["resnet101"]["Sign-OPT"][2000],
        SignOPT_resnet101_5000=result["resnet101"]["Sign-OPT"][5000],
        SignOPT_resnet101_8000=result["resnet101"]["Sign-OPT"][8000],
        SignOPT_resnet101_10000=result["resnet101"]["Sign-OPT"][10000],

        SignOPT_resnext101_300=result["resnext101_64x4d"]["Sign-OPT"][300],
        SignOPT_resnext101_1000=result["resnext101_64x4d"]["Sign-OPT"][1000],
        SignOPT_resnext101_2000=result["resnext101_64x4d"]["Sign-OPT"][2000],
        SignOPT_resnext101_5000=result["resnext101_64x4d"]["Sign-OPT"][5000],
        SignOPT_resnext101_8000=result["resnext101_64x4d"]["Sign-OPT"][8000],
        SignOPT_resnext101_10000=result["resnext101_64x4d"]["Sign-OPT"][10000],

        SVMOPT_inceptionv3_300=result["inceptionv3"]["SVM-OPT"][300],
        SVMOPT_inceptionv3_1000=result["inceptionv3"]["SVM-OPT"][1000],
        SVMOPT_inceptionv3_2000=result["inceptionv3"]["SVM-OPT"][2000],
        SVMOPT_inceptionv3_5000=result["inceptionv3"]["SVM-OPT"][5000],
        SVMOPT_inceptionv3_8000=result["inceptionv3"]["SVM-OPT"][8000],
        SVMOPT_inceptionv3_10000=result["inceptionv3"]["SVM-OPT"][10000],

        SVMOPT_inceptionv4_300=result["inceptionv4"]["SVM-OPT"][300],
        SVMOPT_inceptionv4_1000=result["inceptionv4"]["SVM-OPT"][1000],
        SVMOPT_inceptionv4_2000=result["inceptionv4"]["SVM-OPT"][2000],
        SVMOPT_inceptionv4_5000=result["inceptionv4"]["SVM-OPT"][5000],
        SVMOPT_inceptionv4_8000=result["inceptionv4"]["SVM-OPT"][8000],
        SVMOPT_inceptionv4_10000=result["inceptionv4"]["SVM-OPT"][10000],

        SVMOPT_senet154_300=result["senet154"]["SVM-OPT"][300],
        SVMOPT_senet154_1000=result["senet154"]["SVM-OPT"][1000],
        SVMOPT_senet154_2000=result["senet154"]["SVM-OPT"][2000],
        SVMOPT_senet154_5000=result["senet154"]["SVM-OPT"][5000],
        SVMOPT_senet154_8000=result["senet154"]["SVM-OPT"][8000],
        SVMOPT_senet154_10000=result["senet154"]["SVM-OPT"][10000],

        SVMOPT_resnet101_300=result["resnet101"]["SVM-OPT"][300],
        SVMOPT_resnet101_1000=result["resnet101"]["SVM-OPT"][1000],
        SVMOPT_resnet101_2000=result["resnet101"]["SVM-OPT"][2000],
        SVMOPT_resnet101_5000=result["resnet101"]["SVM-OPT"][5000],
        SVMOPT_resnet101_8000=result["resnet101"]["SVM-OPT"][8000],
        SVMOPT_resnet101_10000=result["resnet101"]["SVM-OPT"][10000],

        SVMOPT_resnext101_300=result["resnext101_64x4d"]["SVM-OPT"][300],
        SVMOPT_resnext101_1000=result["resnext101_64x4d"]["SVM-OPT"][1000],
        SVMOPT_resnext101_2000=result["resnext101_64x4d"]["SVM-OPT"][2000],
        SVMOPT_resnext101_5000=result["resnext101_64x4d"]["SVM-OPT"][5000],
        SVMOPT_resnext101_8000=result["resnext101_64x4d"]["SVM-OPT"][8000],
        SVMOPT_resnext101_10000=result["resnext101_64x4d"]["SVM-OPT"][10000],

        BoundaryAttack_inceptionv3_300='-',
        BoundaryAttack_inceptionv3_1000=result["inceptionv3"]["Boundary Attack"][1000],
        BoundaryAttack_inceptionv3_2000=result["inceptionv3"]["Boundary Attack"][2000],
        BoundaryAttack_inceptionv3_5000=result["inceptionv3"]["Boundary Attack"][5000],
        BoundaryAttack_inceptionv3_8000=result["inceptionv3"]["Boundary Attack"][8000],
        BoundaryAttack_inceptionv3_10000=result["inceptionv3"]["Boundary Attack"][10000],

        BoundaryAttack_inceptionv4_300='-',
        BoundaryAttack_inceptionv4_1000=result["inceptionv4"]["Boundary Attack"][1000],
        BoundaryAttack_inceptionv4_2000=result["inceptionv4"]["Boundary Attack"][2000],
        BoundaryAttack_inceptionv4_5000=result["inceptionv4"]["Boundary Attack"][5000],
        BoundaryAttack_inceptionv4_8000=result["inceptionv4"]["Boundary Attack"][8000],
        BoundaryAttack_inceptionv4_10000=result["inceptionv4"]["Boundary Attack"][10000],

        BoundaryAttack_senet154_300='-',
        BoundaryAttack_senet154_1000=result["senet154"]["Boundary Attack"][1000],
        BoundaryAttack_senet154_2000=result["senet154"]["Boundary Attack"][2000],
        BoundaryAttack_senet154_5000=result["senet154"]["Boundary Attack"][5000],
        BoundaryAttack_senet154_8000=result["senet154"]["Boundary Attack"][8000],
        BoundaryAttack_senet154_10000=result["senet154"]["Boundary Attack"][10000],

        BoundaryAttack_resnet101_300='-',
        BoundaryAttack_resnet101_1000=result["resnet101"]["Boundary Attack"][1000],
        BoundaryAttack_resnet101_2000=result["resnet101"]["Boundary Attack"][2000],
        BoundaryAttack_resnet101_5000=result["resnet101"]["Boundary Attack"][5000],
        BoundaryAttack_resnet101_8000=result["resnet101"]["Boundary Attack"][8000],
        BoundaryAttack_resnet101_10000=result["resnet101"]["Boundary Attack"][10000],

        BoundaryAttack_resnext101_300='-',
        BoundaryAttack_resnext101_1000=result["resnext101_64x4d"]["Boundary Attack"][1000],
        BoundaryAttack_resnext101_2000=result["resnext101_64x4d"]["Boundary Attack"][2000],
        BoundaryAttack_resnext101_5000=result["resnext101_64x4d"]["Boundary Attack"][5000],
        BoundaryAttack_resnext101_8000=result["resnext101_64x4d"]["Boundary Attack"][8000],
        BoundaryAttack_resnext101_10000=result["resnext101_64x4d"]["Boundary Attack"][10000],

        HSJA_inceptionv3_300=result["inceptionv3"]["HopSkipJumpAttack"][300],
        HSJA_inceptionv3_1000=result["inceptionv3"]["HopSkipJumpAttack"][1000],
        HSJA_inceptionv3_2000=result["inceptionv3"]["HopSkipJumpAttack"][2000],
        HSJA_inceptionv3_5000=result["inceptionv3"]["HopSkipJumpAttack"][5000],
        HSJA_inceptionv3_8000=result["inceptionv3"]["HopSkipJumpAttack"][8000],
        HSJA_inceptionv3_10000=result["inceptionv3"]["HopSkipJumpAttack"][10000],

        HSJA_inceptionv4_300=result["inceptionv4"]["HopSkipJumpAttack"][300],
        HSJA_inceptionv4_1000=result["inceptionv4"]["HopSkipJumpAttack"][1000],
        HSJA_inceptionv4_2000=result["inceptionv4"]["HopSkipJumpAttack"][2000],
        HSJA_inceptionv4_5000=result["inceptionv4"]["HopSkipJumpAttack"][5000],
        HSJA_inceptionv4_8000=result["inceptionv4"]["HopSkipJumpAttack"][8000],
        HSJA_inceptionv4_10000=result["inceptionv4"]["HopSkipJumpAttack"][10000],

        HSJA_senet154_300=result["senet154"]["HopSkipJumpAttack"][300],
        HSJA_senet154_1000=result["senet154"]["HopSkipJumpAttack"][1000],
        HSJA_senet154_2000=result["senet154"]["HopSkipJumpAttack"][2000],
        HSJA_senet154_5000=result["senet154"]["HopSkipJumpAttack"][5000],
        HSJA_senet154_8000=result["senet154"]["HopSkipJumpAttack"][8000],
        HSJA_senet154_10000=result["senet154"]["HopSkipJumpAttack"][10000],

        HSJA_resnet101_300=result["resnet101"]["HopSkipJumpAttack"][300],
        HSJA_resnet101_1000=result["resnet101"]["HopSkipJumpAttack"][1000],
        HSJA_resnet101_2000=result["resnet101"]["HopSkipJumpAttack"][2000],
        HSJA_resnet101_5000=result["resnet101"]["HopSkipJumpAttack"][5000],
        HSJA_resnet101_8000=result["resnet101"]["HopSkipJumpAttack"][8000],
        HSJA_resnet101_10000=result["resnet101"]["HopSkipJumpAttack"][10000],

        HSJA_resnext101_300=result["resnext101_64x4d"]["HopSkipJumpAttack"][300],
        HSJA_resnext101_1000=result["resnext101_64x4d"]["HopSkipJumpAttack"][1000],
        HSJA_resnext101_2000=result["resnext101_64x4d"]["HopSkipJumpAttack"][2000],
        HSJA_resnext101_5000=result["resnext101_64x4d"]["HopSkipJumpAttack"][5000],
        HSJA_resnext101_8000=result["resnext101_64x4d"]["HopSkipJumpAttack"][8000],
        HSJA_resnext101_10000=result["resnext101_64x4d"]["HopSkipJumpAttack"][10000],

        Tangent_inceptionv3_300=result["inceptionv3"]["Tangent Attack"][300],
        Tangent_inceptionv3_1000=result["inceptionv3"]["Tangent Attack"][1000],
        Tangent_inceptionv3_2000=result["inceptionv3"]["Tangent Attack"][2000],
        Tangent_inceptionv3_5000=result["inceptionv3"]["Tangent Attack"][5000],
        Tangent_inceptionv3_8000=result["inceptionv3"]["Tangent Attack"][8000],
        Tangent_inceptionv3_10000=result["inceptionv3"]["Tangent Attack"][10000],

        Tangent_inceptionv4_300=result["inceptionv4"]["Tangent Attack"][300],
        Tangent_inceptionv4_1000=result["inceptionv4"]["Tangent Attack"][1000],
        Tangent_inceptionv4_2000=result["inceptionv4"]["Tangent Attack"][2000],
        Tangent_inceptionv4_5000=result["inceptionv4"]["Tangent Attack"][5000],
        Tangent_inceptionv4_8000=result["inceptionv4"]["Tangent Attack"][8000],
        Tangent_inceptionv4_10000=result["inceptionv4"]["Tangent Attack"][10000],

        Tangent_senet154_300=result["senet154"]["Tangent Attack"][300],
        Tangent_senet154_1000=result["senet154"]["Tangent Attack"][1000],
        Tangent_senet154_2000=result["senet154"]["Tangent Attack"][2000],
        Tangent_senet154_5000=result["senet154"]["Tangent Attack"][5000],
        Tangent_senet154_8000=result["senet154"]["Tangent Attack"][8000],
        Tangent_senet154_10000=result["senet154"]["Tangent Attack"][10000],

        Tangent_resnet101_300=result["resnet101"]["Tangent Attack"][300],
        Tangent_resnet101_1000=result["resnet101"]["Tangent Attack"][1000],
        Tangent_resnet101_2000=result["resnet101"]["Tangent Attack"][2000],
        Tangent_resnet101_5000=result["resnet101"]["Tangent Attack"][5000],
        Tangent_resnet101_8000=result["resnet101"]["Tangent Attack"][8000],
        Tangent_resnet101_10000=result["resnet101"]["Tangent Attack"][10000],

        Tangent_resnext101_300=result["resnext101_64x4d"]["Tangent Attack"][300],
        Tangent_resnext101_1000=result["resnext101_64x4d"]["Tangent Attack"][1000],
        Tangent_resnext101_2000=result["resnext101_64x4d"]["Tangent Attack"][2000],
        Tangent_resnext101_5000=result["resnext101_64x4d"]["Tangent Attack"][5000],
        Tangent_resnext101_8000=result["resnext101_64x4d"]["Tangent Attack"][8000],
        Tangent_resnext101_10000=result["resnext101_64x4d"]["Tangent Attack"][10000],
    )
    )

def draw_targeted_first_tables_for_ImageNet_with_300_query(result, targeted_attack_result):
    print("""
                    & Boundary Attack & {targeted_BoundaryAttack_inceptionv3_300} & {targeted_BoundaryAttack_inceptionv3_1000} & {targeted_BoundaryAttack_inceptionv3_2000} & {targeted_BoundaryAttack_inceptionv3_5000} & {targeted_BoundaryAttack_inceptionv3_8000} & {targeted_BoundaryAttack_inceptionv3_10000} & {BoundaryAttack_inceptionv3_300} & {BoundaryAttack_inceptionv3_1000} & {BoundaryAttack_inceptionv3_2000} & {BoundaryAttack_inceptionv3_5000} & {BoundaryAttack_inceptionv3_8000} & {BoundaryAttack_inceptionv3_10000} \\\\
                    & Sign-OPT & {targeted_SignOPT_inceptionv3_300} & {targeted_SignOPT_inceptionv3_1000} & {targeted_SignOPT_inceptionv3_2000} & {targeted_SignOPT_inceptionv3_5000} & {targeted_SignOPT_inceptionv3_8000} & {targeted_SignOPT_inceptionv3_10000} & {SignOPT_inceptionv3_300} & {SignOPT_inceptionv3_1000} & {SignOPT_inceptionv3_2000} & {SignOPT_inceptionv3_5000} & {SignOPT_inceptionv3_8000} & {SignOPT_inceptionv3_10000}\\\\
                    & SVM-OPT &  {targeted_SVMOPT_inceptionv3_300} & {targeted_SVMOPT_inceptionv3_1000} & {targeted_SVMOPT_inceptionv3_2000} & {targeted_SVMOPT_inceptionv3_5000} & {targeted_SVMOPT_inceptionv3_8000} & {targeted_SVMOPT_inceptionv3_10000} & {SVMOPT_inceptionv3_300} & {SVMOPT_inceptionv3_1000} & {SVMOPT_inceptionv3_2000} & {SVMOPT_inceptionv3_5000} & {SVMOPT_inceptionv3_8000} & {SVMOPT_inceptionv3_10000}\\\\
                    & HSJA & {targeted_HSJA_inceptionv3_300} & {targeted_HSJA_inceptionv3_1000} & {targeted_HSJA_inceptionv3_2000} & {targeted_HSJA_inceptionv3_5000} & {targeted_HSJA_inceptionv3_8000} & {targeted_HSJA_inceptionv3_10000} & {HSJA_inceptionv3_300} & {HSJA_inceptionv3_1000} & {HSJA_inceptionv3_2000} & {HSJA_inceptionv3_5000} & {HSJA_inceptionv3_8000} & {HSJA_inceptionv3_10000} \\\\
                    & Ours &  {targeted_Tangent_inceptionv3_300} & {targeted_Tangent_inceptionv3_1000} & {targeted_Tangent_inceptionv3_2000} & {targeted_Tangent_inceptionv3_5000} & {targeted_Tangent_inceptionv3_8000} & {targeted_Tangent_inceptionv3_10000} & {Tangent_inceptionv3_300} & {Tangent_inceptionv3_1000} & {Tangent_inceptionv3_2000} & {Tangent_inceptionv3_5000} & {Tangent_inceptionv3_8000} & {Tangent_inceptionv3_10000}\\\\
                    \\midrule
                    & Boundary Attack & {targeted_BoundaryAttack_inceptionv4_300} & {targeted_BoundaryAttack_inceptionv4_1000} & {targeted_BoundaryAttack_inceptionv4_2000} & {targeted_BoundaryAttack_inceptionv4_5000} & {targeted_BoundaryAttack_inceptionv4_8000} & {targeted_BoundaryAttack_inceptionv4_10000} & {BoundaryAttack_inceptionv4_300} & {BoundaryAttack_inceptionv4_1000} & {BoundaryAttack_inceptionv4_2000} & {BoundaryAttack_inceptionv4_5000} & {BoundaryAttack_inceptionv4_8000} & {BoundaryAttack_inceptionv4_10000} \\\\
                    & Sign-OPT &  {targeted_SignOPT_inceptionv4_300} & {targeted_SignOPT_inceptionv4_1000} & {targeted_SignOPT_inceptionv4_2000} & {targeted_SignOPT_inceptionv4_5000} & {targeted_SignOPT_inceptionv4_8000} & {targeted_SignOPT_inceptionv4_10000} & {SignOPT_inceptionv4_300} & {SignOPT_inceptionv4_1000} & {SignOPT_inceptionv4_2000} & {SignOPT_inceptionv4_5000} & {SignOPT_inceptionv4_8000} & {SignOPT_inceptionv4_10000}\\\\
                    & SVM-OPT & {targeted_SVMOPT_inceptionv4_300} & {targeted_SVMOPT_inceptionv4_1000} & {targeted_SVMOPT_inceptionv4_2000} & {targeted_SVMOPT_inceptionv4_5000} & {targeted_SVMOPT_inceptionv4_8000} & {targeted_SVMOPT_inceptionv4_10000} & {SVMOPT_inceptionv4_300} & {SVMOPT_inceptionv4_1000} & {SVMOPT_inceptionv4_2000} & {SVMOPT_inceptionv4_5000} & {SVMOPT_inceptionv4_8000} & {SVMOPT_inceptionv4_10000} \\\\
                    & HSJA &  {targeted_HSJA_inceptionv4_300} & {targeted_HSJA_inceptionv4_1000} & {targeted_HSJA_inceptionv4_2000} & {targeted_HSJA_inceptionv4_5000} & {targeted_HSJA_inceptionv4_8000} & {targeted_HSJA_inceptionv4_10000} & {HSJA_inceptionv4_300} & {HSJA_inceptionv4_1000} & {HSJA_inceptionv4_2000} & {HSJA_inceptionv4_5000} & {HSJA_inceptionv4_8000} & {HSJA_inceptionv4_10000}\\\\
                    & Ours & {targeted_Tangent_inceptionv4_300} & {targeted_Tangent_inceptionv4_1000} & {targeted_Tangent_inceptionv4_2000} & {targeted_Tangent_inceptionv4_5000} & {targeted_Tangent_inceptionv4_8000} & {targeted_Tangent_inceptionv4_10000} & {Tangent_inceptionv4_300} & {Tangent_inceptionv4_1000} & {Tangent_inceptionv4_2000} & {Tangent_inceptionv4_5000} & {Tangent_inceptionv4_8000} & {Tangent_inceptionv4_10000} \\\\
                    \\midrule
                    & Boundary Attack & {targeted_BoundaryAttack_senet154_300} & {targeted_BoundaryAttack_senet154_1000} & {targeted_BoundaryAttack_senet154_2000} & {targeted_BoundaryAttack_senet154_5000} & {targeted_BoundaryAttack_senet154_8000} & {targeted_BoundaryAttack_senet154_10000} & {BoundaryAttack_senet154_300} & {BoundaryAttack_senet154_1000} & {BoundaryAttack_senet154_2000} & {BoundaryAttack_senet154_5000} & {BoundaryAttack_senet154_8000} & {BoundaryAttack_senet154_10000} \\\\
                    & Sign-OPT &  {targeted_SignOPT_senet154_300} & {targeted_SignOPT_senet154_1000} & {targeted_SignOPT_senet154_2000} & {targeted_SignOPT_senet154_5000} & {targeted_SignOPT_senet154_8000} & {targeted_SignOPT_senet154_10000}  & {SignOPT_senet154_300} & {SignOPT_senet154_1000} & {SignOPT_senet154_2000} & {SignOPT_senet154_5000} & {SignOPT_senet154_8000} & {SignOPT_senet154_10000}\\\\
                    & SVM-OPT & {targeted_SVMOPT_senet154_300} & {targeted_SVMOPT_senet154_1000} & {targeted_SVMOPT_senet154_2000} & {targeted_SVMOPT_senet154_5000} & {targeted_SVMOPT_senet154_8000} & {targeted_SVMOPT_senet154_10000} & {SVMOPT_senet154_300} & {SVMOPT_senet154_1000} & {SVMOPT_senet154_2000} & {SVMOPT_senet154_5000} & {SVMOPT_senet154_8000} & {SVMOPT_senet154_10000} \\\\
                    & HSJA &  {targeted_HSJA_senet154_300} & {targeted_HSJA_senet154_1000} & {targeted_HSJA_senet154_2000} & {targeted_HSJA_senet154_5000} & {targeted_HSJA_senet154_8000} & {targeted_HSJA_senet154_10000} & {HSJA_senet154_300} & {HSJA_senet154_1000} & {HSJA_senet154_2000} & {HSJA_senet154_5000} & {HSJA_senet154_8000} & {HSJA_senet154_10000}\\\\
                    & Ours &  {targeted_Tangent_senet154_300} & {targeted_Tangent_senet154_1000} & {targeted_Tangent_senet154_2000} & {targeted_Tangent_senet154_5000} & {targeted_Tangent_senet154_8000} & {targeted_Tangent_senet154_10000} & {Tangent_senet154_300} & {Tangent_senet154_1000} & {Tangent_senet154_2000} & {Tangent_senet154_5000} & {Tangent_senet154_8000} & {Tangent_senet154_10000}\\\\
                    \\midrule
                    & Boundary Attack & {targeted_BoundaryAttack_resnet101_300} & {targeted_BoundaryAttack_resnet101_1000} & {targeted_BoundaryAttack_resnet101_2000} & {targeted_BoundaryAttack_resnet101_5000} & {targeted_BoundaryAttack_resnet101_8000} & {targeted_BoundaryAttack_resnet101_10000} & {BoundaryAttack_resnet101_300} & {BoundaryAttack_resnet101_1000} & {BoundaryAttack_resnet101_2000} & {BoundaryAttack_resnet101_5000} & {BoundaryAttack_resnet101_8000} & {BoundaryAttack_resnet101_10000} \\\\
                   & Sign-OPT & {targeted_SignOPT_resnet101_300} & {targeted_SignOPT_resnet101_1000} & {targeted_SignOPT_resnet101_2000} & {targeted_SignOPT_resnet101_5000} & {targeted_SignOPT_resnet101_8000} & {targeted_SignOPT_resnet101_10000} & {SignOPT_resnet101_300} & {SignOPT_resnet101_1000} & {SignOPT_resnet101_2000} & {SignOPT_resnet101_5000} & {SignOPT_resnet101_8000} & {SignOPT_resnet101_10000} \\\\
                    & SVM-OPT &  {targeted_SVMOPT_resnet101_300} & {targeted_SVMOPT_resnet101_1000} & {targeted_SVMOPT_resnet101_2000} & {targeted_SVMOPT_resnet101_5000} & {targeted_SVMOPT_resnet101_8000} & {targeted_SVMOPT_resnet101_10000} & {SVMOPT_resnet101_300} & {SVMOPT_resnet101_1000} & {SVMOPT_resnet101_2000} & {SVMOPT_resnet101_5000} & {SVMOPT_resnet101_8000} & {SVMOPT_resnet101_10000}\\\\
                    & HSJA & {targeted_HSJA_resnet101_300} & {targeted_HSJA_resnet101_1000} & {targeted_HSJA_resnet101_2000} & {targeted_HSJA_resnet101_5000} & {targeted_HSJA_resnet101_8000} & {targeted_HSJA_resnet101_10000} & {HSJA_resnet101_300} & {HSJA_resnet101_1000} & {HSJA_resnet101_2000} & {HSJA_resnet101_5000} & {HSJA_resnet101_8000} & {HSJA_resnet101_10000} \\\\
                    & Ours & {targeted_Tangent_resnet101_300} & {targeted_Tangent_resnet101_1000} & {targeted_Tangent_resnet101_2000} & {targeted_Tangent_resnet101_5000} & {targeted_Tangent_resnet101_8000} & {targeted_Tangent_resnet101_10000} & {Tangent_resnet101_300} & {Tangent_resnet101_1000} & {Tangent_resnet101_2000} & {Tangent_resnet101_5000} & {Tangent_resnet101_8000} & {Tangent_resnet101_10000} \\\\
                        """.format(

        SignOPT_inceptionv3_300=result["inceptionv3"]["Sign-OPT"][300],
        SignOPT_inceptionv3_1000=result["inceptionv3"]["Sign-OPT"][1000],
        SignOPT_inceptionv3_2000=result["inceptionv3"]["Sign-OPT"][2000],
        SignOPT_inceptionv3_5000=result["inceptionv3"]["Sign-OPT"][5000],
        SignOPT_inceptionv3_8000=result["inceptionv3"]["Sign-OPT"][8000],
        SignOPT_inceptionv3_10000=result["inceptionv3"]["Sign-OPT"][10000],

        SignOPT_inceptionv4_300=result["inceptionv4"]["Sign-OPT"][300],
        SignOPT_inceptionv4_1000=result["inceptionv4"]["Sign-OPT"][1000],
        SignOPT_inceptionv4_2000=result["inceptionv4"]["Sign-OPT"][2000],
        SignOPT_inceptionv4_5000=result["inceptionv4"]["Sign-OPT"][5000],
        SignOPT_inceptionv4_8000=result["inceptionv4"]["Sign-OPT"][8000],
        SignOPT_inceptionv4_10000=result["inceptionv4"]["Sign-OPT"][10000],

        SignOPT_senet154_300=result["senet154"]["Sign-OPT"][300],
        SignOPT_senet154_1000=result["senet154"]["Sign-OPT"][1000],
        SignOPT_senet154_2000=result["senet154"]["Sign-OPT"][2000],
        SignOPT_senet154_5000=result["senet154"]["Sign-OPT"][5000],
        SignOPT_senet154_8000=result["senet154"]["Sign-OPT"][8000],
        SignOPT_senet154_10000=result["senet154"]["Sign-OPT"][10000],

        SignOPT_resnet101_300=result["resnet101"]["Sign-OPT"][300],
        SignOPT_resnet101_1000=result["resnet101"]["Sign-OPT"][1000],
        SignOPT_resnet101_2000=result["resnet101"]["Sign-OPT"][2000],
        SignOPT_resnet101_5000=result["resnet101"]["Sign-OPT"][5000],
        SignOPT_resnet101_8000=result["resnet101"]["Sign-OPT"][8000],
        SignOPT_resnet101_10000=result["resnet101"]["Sign-OPT"][10000],


        SVMOPT_inceptionv3_300=result["inceptionv3"]["SVM-OPT"][300],
        SVMOPT_inceptionv3_1000=result["inceptionv3"]["SVM-OPT"][1000],
        SVMOPT_inceptionv3_2000=result["inceptionv3"]["SVM-OPT"][2000],
        SVMOPT_inceptionv3_5000=result["inceptionv3"]["SVM-OPT"][5000],
        SVMOPT_inceptionv3_8000=result["inceptionv3"]["SVM-OPT"][8000],
        SVMOPT_inceptionv3_10000=result["inceptionv3"]["SVM-OPT"][10000],

        SVMOPT_inceptionv4_300=result["inceptionv4"]["SVM-OPT"][300],
        SVMOPT_inceptionv4_1000=result["inceptionv4"]["SVM-OPT"][1000],
        SVMOPT_inceptionv4_2000=result["inceptionv4"]["SVM-OPT"][2000],
        SVMOPT_inceptionv4_5000=result["inceptionv4"]["SVM-OPT"][5000],
        SVMOPT_inceptionv4_8000=result["inceptionv4"]["SVM-OPT"][8000],
        SVMOPT_inceptionv4_10000=result["inceptionv4"]["SVM-OPT"][10000],

        SVMOPT_senet154_300=result["senet154"]["SVM-OPT"][300],
        SVMOPT_senet154_1000=result["senet154"]["SVM-OPT"][1000],
        SVMOPT_senet154_2000=result["senet154"]["SVM-OPT"][2000],
        SVMOPT_senet154_5000=result["senet154"]["SVM-OPT"][5000],
        SVMOPT_senet154_8000=result["senet154"]["SVM-OPT"][8000],
        SVMOPT_senet154_10000=result["senet154"]["SVM-OPT"][10000],

        SVMOPT_resnet101_300=result["resnet101"]["SVM-OPT"][300],
        SVMOPT_resnet101_1000=result["resnet101"]["SVM-OPT"][1000],
        SVMOPT_resnet101_2000=result["resnet101"]["SVM-OPT"][2000],
        SVMOPT_resnet101_5000=result["resnet101"]["SVM-OPT"][5000],
        SVMOPT_resnet101_8000=result["resnet101"]["SVM-OPT"][8000],
        SVMOPT_resnet101_10000=result["resnet101"]["SVM-OPT"][10000],


        BoundaryAttack_inceptionv3_300=result["inceptionv3"]["Boundary Attack"][300],
        BoundaryAttack_inceptionv3_1000=result["inceptionv3"]["Boundary Attack"][1000],
        BoundaryAttack_inceptionv3_2000=result["inceptionv3"]["Boundary Attack"][2000],
        BoundaryAttack_inceptionv3_5000=result["inceptionv3"]["Boundary Attack"][5000],
        BoundaryAttack_inceptionv3_8000=result["inceptionv3"]["Boundary Attack"][8000],
        BoundaryAttack_inceptionv3_10000=result["inceptionv3"]["Boundary Attack"][10000],

        BoundaryAttack_inceptionv4_300=result["inceptionv4"]["Boundary Attack"][300],
        BoundaryAttack_inceptionv4_1000=result["inceptionv4"]["Boundary Attack"][1000],
        BoundaryAttack_inceptionv4_2000=result["inceptionv4"]["Boundary Attack"][2000],
        BoundaryAttack_inceptionv4_5000=result["inceptionv4"]["Boundary Attack"][5000],
        BoundaryAttack_inceptionv4_8000=result["inceptionv4"]["Boundary Attack"][8000],
        BoundaryAttack_inceptionv4_10000=result["inceptionv4"]["Boundary Attack"][10000],

        BoundaryAttack_senet154_300=result["senet154"]["Boundary Attack"][300],
        BoundaryAttack_senet154_1000=result["senet154"]["Boundary Attack"][1000],
        BoundaryAttack_senet154_2000=result["senet154"]["Boundary Attack"][2000],
        BoundaryAttack_senet154_5000=result["senet154"]["Boundary Attack"][5000],
        BoundaryAttack_senet154_8000=result["senet154"]["Boundary Attack"][8000],
        BoundaryAttack_senet154_10000=result["senet154"]["Boundary Attack"][10000],

        BoundaryAttack_resnet101_300=result["resnet101"]["Boundary Attack"][300],
        BoundaryAttack_resnet101_1000=result["resnet101"]["Boundary Attack"][1000],
        BoundaryAttack_resnet101_2000=result["resnet101"]["Boundary Attack"][2000],
        BoundaryAttack_resnet101_5000=result["resnet101"]["Boundary Attack"][5000],
        BoundaryAttack_resnet101_8000=result["resnet101"]["Boundary Attack"][8000],
        BoundaryAttack_resnet101_10000=result["resnet101"]["Boundary Attack"][10000],


        HSJA_inceptionv3_300=result["inceptionv3"]["HopSkipJumpAttack"][300],
        HSJA_inceptionv3_1000=result["inceptionv3"]["HopSkipJumpAttack"][1000],
        HSJA_inceptionv3_2000=result["inceptionv3"]["HopSkipJumpAttack"][2000],
        HSJA_inceptionv3_5000=result["inceptionv3"]["HopSkipJumpAttack"][5000],
        HSJA_inceptionv3_8000=result["inceptionv3"]["HopSkipJumpAttack"][8000],
        HSJA_inceptionv3_10000=result["inceptionv3"]["HopSkipJumpAttack"][10000],

        HSJA_inceptionv4_300=result["inceptionv4"]["HopSkipJumpAttack"][300],
        HSJA_inceptionv4_1000=result["inceptionv4"]["HopSkipJumpAttack"][1000],
        HSJA_inceptionv4_2000=result["inceptionv4"]["HopSkipJumpAttack"][2000],
        HSJA_inceptionv4_5000=result["inceptionv4"]["HopSkipJumpAttack"][5000],
        HSJA_inceptionv4_8000=result["inceptionv4"]["HopSkipJumpAttack"][8000],
        HSJA_inceptionv4_10000=result["inceptionv4"]["HopSkipJumpAttack"][10000],

        HSJA_senet154_300=result["senet154"]["HopSkipJumpAttack"][300],
        HSJA_senet154_1000=result["senet154"]["HopSkipJumpAttack"][1000],
        HSJA_senet154_2000=result["senet154"]["HopSkipJumpAttack"][2000],
        HSJA_senet154_5000=result["senet154"]["HopSkipJumpAttack"][5000],
        HSJA_senet154_8000=result["senet154"]["HopSkipJumpAttack"][8000],
        HSJA_senet154_10000=result["senet154"]["HopSkipJumpAttack"][10000],

        HSJA_resnet101_300=result["resnet101"]["HopSkipJumpAttack"][300],
        HSJA_resnet101_1000=result["resnet101"]["HopSkipJumpAttack"][1000],
        HSJA_resnet101_2000=result["resnet101"]["HopSkipJumpAttack"][2000],
        HSJA_resnet101_5000=result["resnet101"]["HopSkipJumpAttack"][5000],
        HSJA_resnet101_8000=result["resnet101"]["HopSkipJumpAttack"][8000],
        HSJA_resnet101_10000=result["resnet101"]["HopSkipJumpAttack"][10000],



        Tangent_inceptionv3_300=result["inceptionv3"]["Tangent Attack"][300],
        Tangent_inceptionv3_1000=result["inceptionv3"]["Tangent Attack"][1000],
        Tangent_inceptionv3_2000=result["inceptionv3"]["Tangent Attack"][2000],
        Tangent_inceptionv3_5000=result["inceptionv3"]["Tangent Attack"][5000],
        Tangent_inceptionv3_8000=result["inceptionv3"]["Tangent Attack"][8000],
        Tangent_inceptionv3_10000=result["inceptionv3"]["Tangent Attack"][10000],

        Tangent_inceptionv4_300=result["inceptionv4"]["Tangent Attack"][300],
        Tangent_inceptionv4_1000=result["inceptionv4"]["Tangent Attack"][1000],
        Tangent_inceptionv4_2000=result["inceptionv4"]["Tangent Attack"][2000],
        Tangent_inceptionv4_5000=result["inceptionv4"]["Tangent Attack"][5000],
        Tangent_inceptionv4_8000=result["inceptionv4"]["Tangent Attack"][8000],
        Tangent_inceptionv4_10000=result["inceptionv4"]["Tangent Attack"][10000],

        Tangent_senet154_300=result["senet154"]["Tangent Attack"][300],
        Tangent_senet154_1000=result["senet154"]["Tangent Attack"][1000],
        Tangent_senet154_2000=result["senet154"]["Tangent Attack"][2000],
        Tangent_senet154_5000=result["senet154"]["Tangent Attack"][5000],
        Tangent_senet154_8000=result["senet154"]["Tangent Attack"][8000],
        Tangent_senet154_10000=result["senet154"]["Tangent Attack"][10000],

        Tangent_resnet101_300=result["resnet101"]["Tangent Attack"][300],
        Tangent_resnet101_1000=result["resnet101"]["Tangent Attack"][1000],
        Tangent_resnet101_2000=result["resnet101"]["Tangent Attack"][2000],
        Tangent_resnet101_5000=result["resnet101"]["Tangent Attack"][5000],
        Tangent_resnet101_8000=result["resnet101"]["Tangent Attack"][8000],
        Tangent_resnet101_10000=result["resnet101"]["Tangent Attack"][10000],



        targeted_SignOPT_inceptionv3_300=targeted_attack_result["inceptionv3"]["Sign-OPT"][300],
        targeted_SignOPT_inceptionv3_1000=targeted_attack_result["inceptionv3"]["Sign-OPT"][1000],
        targeted_SignOPT_inceptionv3_2000=targeted_attack_result["inceptionv3"]["Sign-OPT"][2000],
        targeted_SignOPT_inceptionv3_5000=targeted_attack_result["inceptionv3"]["Sign-OPT"][5000],
        targeted_SignOPT_inceptionv3_8000=targeted_attack_result["inceptionv3"]["Sign-OPT"][8000],
        targeted_SignOPT_inceptionv3_10000=targeted_attack_result["inceptionv3"]["Sign-OPT"][10000],
        targeted_SignOPT_inceptionv4_300=targeted_attack_result["inceptionv4"]["Sign-OPT"][300],
        targeted_SignOPT_inceptionv4_1000=targeted_attack_result["inceptionv4"]["Sign-OPT"][1000],
        targeted_SignOPT_inceptionv4_2000=targeted_attack_result["inceptionv4"]["Sign-OPT"][2000],
        targeted_SignOPT_inceptionv4_5000=targeted_attack_result["inceptionv4"]["Sign-OPT"][5000],
        targeted_SignOPT_inceptionv4_8000=targeted_attack_result["inceptionv4"]["Sign-OPT"][8000],
        targeted_SignOPT_inceptionv4_10000=targeted_attack_result["inceptionv4"]["Sign-OPT"][10000],
        targeted_SignOPT_senet154_300=targeted_attack_result["senet154"]["Sign-OPT"][300],
        targeted_SignOPT_senet154_1000=targeted_attack_result["senet154"]["Sign-OPT"][1000],
        targeted_SignOPT_senet154_2000=targeted_attack_result["senet154"]["Sign-OPT"][2000],
        targeted_SignOPT_senet154_5000=targeted_attack_result["senet154"]["Sign-OPT"][5000],
        targeted_SignOPT_senet154_8000=targeted_attack_result["senet154"]["Sign-OPT"][8000],
        targeted_SignOPT_senet154_10000=targeted_attack_result["senet154"]["Sign-OPT"][10000],
        targeted_SignOPT_resnet101_300=targeted_attack_result["resnet101"]["Sign-OPT"][300],
        targeted_SignOPT_resnet101_1000=targeted_attack_result["resnet101"]["Sign-OPT"][1000],
        targeted_SignOPT_resnet101_2000=targeted_attack_result["resnet101"]["Sign-OPT"][2000],
        targeted_SignOPT_resnet101_5000=targeted_attack_result["resnet101"]["Sign-OPT"][5000],
        targeted_SignOPT_resnet101_8000=targeted_attack_result["resnet101"]["Sign-OPT"][8000],
        targeted_SignOPT_resnet101_10000=targeted_attack_result["resnet101"]["Sign-OPT"][10000],


        targeted_SVMOPT_inceptionv3_300=targeted_attack_result["inceptionv3"]["SVM-OPT"][300],
        targeted_SVMOPT_inceptionv3_1000=targeted_attack_result["inceptionv3"]["SVM-OPT"][1000],
        targeted_SVMOPT_inceptionv3_2000=targeted_attack_result["inceptionv3"]["SVM-OPT"][2000],
        targeted_SVMOPT_inceptionv3_5000=targeted_attack_result["inceptionv3"]["SVM-OPT"][5000],
        targeted_SVMOPT_inceptionv3_8000=targeted_attack_result["inceptionv3"]["SVM-OPT"][8000],
        targeted_SVMOPT_inceptionv3_10000=targeted_attack_result["inceptionv3"]["SVM-OPT"][10000],

        targeted_SVMOPT_inceptionv4_300=targeted_attack_result["inceptionv4"]["SVM-OPT"][300],
        targeted_SVMOPT_inceptionv4_1000=targeted_attack_result["inceptionv4"]["SVM-OPT"][1000],
        targeted_SVMOPT_inceptionv4_2000=targeted_attack_result["inceptionv4"]["SVM-OPT"][2000],
        targeted_SVMOPT_inceptionv4_5000=targeted_attack_result["inceptionv4"]["SVM-OPT"][5000],
        targeted_SVMOPT_inceptionv4_8000=targeted_attack_result["inceptionv4"]["SVM-OPT"][8000],
        targeted_SVMOPT_inceptionv4_10000=targeted_attack_result["inceptionv4"]["SVM-OPT"][10000],

        targeted_SVMOPT_senet154_300=targeted_attack_result["senet154"]["SVM-OPT"][300],
        targeted_SVMOPT_senet154_1000=targeted_attack_result["senet154"]["SVM-OPT"][1000],
        targeted_SVMOPT_senet154_2000=targeted_attack_result["senet154"]["SVM-OPT"][2000],
        targeted_SVMOPT_senet154_5000=targeted_attack_result["senet154"]["SVM-OPT"][5000],
        targeted_SVMOPT_senet154_8000=targeted_attack_result["senet154"]["SVM-OPT"][8000],
        targeted_SVMOPT_senet154_10000=targeted_attack_result["senet154"]["SVM-OPT"][10000],

        targeted_SVMOPT_resnet101_300=targeted_attack_result["resnet101"]["SVM-OPT"][300],
        targeted_SVMOPT_resnet101_1000=targeted_attack_result["resnet101"]["SVM-OPT"][1000],
        targeted_SVMOPT_resnet101_2000=targeted_attack_result["resnet101"]["SVM-OPT"][2000],
        targeted_SVMOPT_resnet101_5000=targeted_attack_result["resnet101"]["SVM-OPT"][5000],
        targeted_SVMOPT_resnet101_8000=targeted_attack_result["resnet101"]["SVM-OPT"][8000],
        targeted_SVMOPT_resnet101_10000=targeted_attack_result["resnet101"]["SVM-OPT"][10000],


        targeted_BoundaryAttack_inceptionv3_300=targeted_attack_result["inceptionv3"]["Boundary Attack"][300],
        targeted_BoundaryAttack_inceptionv3_1000=targeted_attack_result["inceptionv3"]["Boundary Attack"][1000],
        targeted_BoundaryAttack_inceptionv3_2000=targeted_attack_result["inceptionv3"]["Boundary Attack"][2000],
        targeted_BoundaryAttack_inceptionv3_5000=targeted_attack_result["inceptionv3"]["Boundary Attack"][5000],
        targeted_BoundaryAttack_inceptionv3_8000=targeted_attack_result["inceptionv3"]["Boundary Attack"][8000],
        targeted_BoundaryAttack_inceptionv3_10000=targeted_attack_result["inceptionv3"]["Boundary Attack"][10000],

        targeted_BoundaryAttack_inceptionv4_300=targeted_attack_result["inceptionv4"]["Boundary Attack"][300],
        targeted_BoundaryAttack_inceptionv4_1000=targeted_attack_result["inceptionv4"]["Boundary Attack"][1000],
        targeted_BoundaryAttack_inceptionv4_2000=targeted_attack_result["inceptionv4"]["Boundary Attack"][2000],
        targeted_BoundaryAttack_inceptionv4_5000=targeted_attack_result["inceptionv4"]["Boundary Attack"][5000],
        targeted_BoundaryAttack_inceptionv4_8000=targeted_attack_result["inceptionv4"]["Boundary Attack"][8000],
        targeted_BoundaryAttack_inceptionv4_10000=targeted_attack_result["inceptionv4"]["Boundary Attack"][10000],

        targeted_BoundaryAttack_senet154_300=targeted_attack_result["senet154"]["Boundary Attack"][300],
        targeted_BoundaryAttack_senet154_1000=targeted_attack_result["senet154"]["Boundary Attack"][1000],
        targeted_BoundaryAttack_senet154_2000=targeted_attack_result["senet154"]["Boundary Attack"][2000],
        targeted_BoundaryAttack_senet154_5000=targeted_attack_result["senet154"]["Boundary Attack"][5000],
        targeted_BoundaryAttack_senet154_8000=targeted_attack_result["senet154"]["Boundary Attack"][8000],
        targeted_BoundaryAttack_senet154_10000=targeted_attack_result["senet154"]["Boundary Attack"][10000],

        targeted_BoundaryAttack_resnet101_300=targeted_attack_result["resnet101"]["Boundary Attack"][300],
        targeted_BoundaryAttack_resnet101_1000=targeted_attack_result["resnet101"]["Boundary Attack"][1000],
        targeted_BoundaryAttack_resnet101_2000=targeted_attack_result["resnet101"]["Boundary Attack"][2000],
        targeted_BoundaryAttack_resnet101_5000=targeted_attack_result["resnet101"]["Boundary Attack"][5000],
        targeted_BoundaryAttack_resnet101_8000=targeted_attack_result["resnet101"]["Boundary Attack"][8000],
        targeted_BoundaryAttack_resnet101_10000=targeted_attack_result["resnet101"]["Boundary Attack"][10000],


        targeted_HSJA_inceptionv3_300=targeted_attack_result["inceptionv3"]["HopSkipJumpAttack"][300],
        targeted_HSJA_inceptionv3_1000=targeted_attack_result["inceptionv3"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_inceptionv3_2000=targeted_attack_result["inceptionv3"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_inceptionv3_5000=targeted_attack_result["inceptionv3"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_inceptionv3_8000=targeted_attack_result["inceptionv3"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_inceptionv3_10000=targeted_attack_result["inceptionv3"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_inceptionv4_300=targeted_attack_result["inceptionv4"]["HopSkipJumpAttack"][300],
        targeted_HSJA_inceptionv4_1000=targeted_attack_result["inceptionv4"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_inceptionv4_2000=targeted_attack_result["inceptionv4"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_inceptionv4_5000=targeted_attack_result["inceptionv4"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_inceptionv4_8000=targeted_attack_result["inceptionv4"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_inceptionv4_10000=targeted_attack_result["inceptionv4"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_senet154_300=targeted_attack_result["senet154"]["HopSkipJumpAttack"][300],
        targeted_HSJA_senet154_1000=targeted_attack_result["senet154"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_senet154_2000=targeted_attack_result["senet154"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_senet154_5000=targeted_attack_result["senet154"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_senet154_8000=targeted_attack_result["senet154"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_senet154_10000=targeted_attack_result["senet154"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_resnet101_300=targeted_attack_result["resnet101"]["HopSkipJumpAttack"][300],
        targeted_HSJA_resnet101_1000=targeted_attack_result["resnet101"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_resnet101_2000=targeted_attack_result["resnet101"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_resnet101_5000=targeted_attack_result["resnet101"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_resnet101_8000=targeted_attack_result["resnet101"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_resnet101_10000=targeted_attack_result["resnet101"]["HopSkipJumpAttack"][10000],


        targeted_Tangent_inceptionv3_300=targeted_attack_result["inceptionv3"]["Tangent Attack"][300],
        targeted_Tangent_inceptionv3_1000=targeted_attack_result["inceptionv3"]["Tangent Attack"][1000],
        targeted_Tangent_inceptionv3_2000=targeted_attack_result["inceptionv3"]["Tangent Attack"][2000],
        targeted_Tangent_inceptionv3_5000=targeted_attack_result["inceptionv3"]["Tangent Attack"][5000],
        targeted_Tangent_inceptionv3_8000=targeted_attack_result["inceptionv3"]["Tangent Attack"][8000],
        targeted_Tangent_inceptionv3_10000=targeted_attack_result["inceptionv3"]["Tangent Attack"][10000],

        targeted_Tangent_inceptionv4_300=targeted_attack_result["inceptionv4"]["Tangent Attack"][300],
        targeted_Tangent_inceptionv4_1000=targeted_attack_result["inceptionv4"]["Tangent Attack"][1000],
        targeted_Tangent_inceptionv4_2000=targeted_attack_result["inceptionv4"]["Tangent Attack"][2000],
        targeted_Tangent_inceptionv4_5000=targeted_attack_result["inceptionv4"]["Tangent Attack"][5000],
        targeted_Tangent_inceptionv4_8000=targeted_attack_result["inceptionv4"]["Tangent Attack"][8000],
        targeted_Tangent_inceptionv4_10000=targeted_attack_result["inceptionv4"]["Tangent Attack"][10000],

        targeted_Tangent_senet154_300=targeted_attack_result["senet154"]["Tangent Attack"][300],
        targeted_Tangent_senet154_1000=targeted_attack_result["senet154"]["Tangent Attack"][1000],
        targeted_Tangent_senet154_2000=targeted_attack_result["senet154"]["Tangent Attack"][2000],
        targeted_Tangent_senet154_5000=targeted_attack_result["senet154"]["Tangent Attack"][5000],
        targeted_Tangent_senet154_8000=targeted_attack_result["senet154"]["Tangent Attack"][8000],
        targeted_Tangent_senet154_10000=targeted_attack_result["senet154"]["Tangent Attack"][10000],

        targeted_Tangent_resnet101_300=targeted_attack_result["resnet101"]["Tangent Attack"][300],
        targeted_Tangent_resnet101_1000=targeted_attack_result["resnet101"]["Tangent Attack"][1000],
        targeted_Tangent_resnet101_2000=targeted_attack_result["resnet101"]["Tangent Attack"][2000],
        targeted_Tangent_resnet101_5000=targeted_attack_result["resnet101"]["Tangent Attack"][5000],
        targeted_Tangent_resnet101_8000=targeted_attack_result["resnet101"]["Tangent Attack"][8000],
        targeted_Tangent_resnet101_10000=targeted_attack_result["resnet101"]["Tangent Attack"][10000],


    )
    )

def draw_wide_tables_for_ImageNet_with_300_query(result, targeted_attack_result):
    print("""
                    & Boundary Attack & {BoundaryAttack_inceptionv3_300} & {BoundaryAttack_inceptionv3_1000} & {BoundaryAttack_inceptionv3_2000} & {BoundaryAttack_inceptionv3_5000} & {BoundaryAttack_inceptionv3_8000} & {BoundaryAttack_inceptionv3_10000} & {targeted_BoundaryAttack_inceptionv3_300} & {targeted_BoundaryAttack_inceptionv3_1000} & {targeted_BoundaryAttack_inceptionv3_2000} & {targeted_BoundaryAttack_inceptionv3_5000} & {targeted_BoundaryAttack_inceptionv3_8000} & {targeted_BoundaryAttack_inceptionv3_10000} \\\\
                    & Sign-OPT & {SignOPT_inceptionv3_300} & {SignOPT_inceptionv3_1000} & {SignOPT_inceptionv3_2000} & {SignOPT_inceptionv3_5000} & {SignOPT_inceptionv3_8000} & {SignOPT_inceptionv3_10000} & {targeted_SignOPT_inceptionv3_300} & {targeted_SignOPT_inceptionv3_1000} & {targeted_SignOPT_inceptionv3_2000} & {targeted_SignOPT_inceptionv3_5000} & {targeted_SignOPT_inceptionv3_8000} & {targeted_SignOPT_inceptionv3_10000} \\\\
                    & SVM-OPT & {SVMOPT_inceptionv3_300} & {SVMOPT_inceptionv3_1000} & {SVMOPT_inceptionv3_2000} & {SVMOPT_inceptionv3_5000} & {SVMOPT_inceptionv3_8000} & {SVMOPT_inceptionv3_10000} & {targeted_SVMOPT_inceptionv3_300} & {targeted_SVMOPT_inceptionv3_1000} & {targeted_SVMOPT_inceptionv3_2000} & {targeted_SVMOPT_inceptionv3_5000} & {targeted_SVMOPT_inceptionv3_8000} & {targeted_SVMOPT_inceptionv3_10000} \\\\
                    & HopSkipJumpAttack & {HSJA_inceptionv3_300} & {HSJA_inceptionv3_1000} & {HSJA_inceptionv3_2000} & {HSJA_inceptionv3_5000} & {HSJA_inceptionv3_8000} & {HSJA_inceptionv3_10000} & {targeted_HSJA_inceptionv3_300} & {targeted_HSJA_inceptionv3_1000} & {targeted_HSJA_inceptionv3_2000} & {targeted_HSJA_inceptionv3_5000} & {targeted_HSJA_inceptionv3_8000} & {targeted_HSJA_inceptionv3_10000} \\\\
                    & Tangent Attack (ours) & {Tangent_inceptionv3_300} & {Tangent_inceptionv3_1000} & {Tangent_inceptionv3_2000} & {Tangent_inceptionv3_5000} & {Tangent_inceptionv3_8000} & {Tangent_inceptionv3_10000} & {targeted_Tangent_inceptionv3_300} & {targeted_Tangent_inceptionv3_1000} & {targeted_Tangent_inceptionv3_2000} & {targeted_Tangent_inceptionv3_5000} & {targeted_Tangent_inceptionv3_8000} & {targeted_Tangent_inceptionv3_10000} \\\\
                    \\midrule
                    & Boundary Attack & {BoundaryAttack_inceptionv4_300} & {BoundaryAttack_inceptionv4_1000} & {BoundaryAttack_inceptionv4_2000} & {BoundaryAttack_inceptionv4_5000} & {BoundaryAttack_inceptionv4_8000} & {BoundaryAttack_inceptionv4_10000} & {targeted_BoundaryAttack_inceptionv4_300} & {targeted_BoundaryAttack_inceptionv4_1000} & {targeted_BoundaryAttack_inceptionv4_2000} & {targeted_BoundaryAttack_inceptionv4_5000} & {targeted_BoundaryAttack_inceptionv4_8000} & {targeted_BoundaryAttack_inceptionv4_10000} \\\\
                    & Sign-OPT & {SignOPT_inceptionv4_300} & {SignOPT_inceptionv4_1000} & {SignOPT_inceptionv4_2000} & {SignOPT_inceptionv4_5000} & {SignOPT_inceptionv4_8000} & {SignOPT_inceptionv4_10000} & {targeted_SignOPT_inceptionv4_300} & {targeted_SignOPT_inceptionv4_1000} & {targeted_SignOPT_inceptionv4_2000} & {targeted_SignOPT_inceptionv4_5000} & {targeted_SignOPT_inceptionv4_8000} & {targeted_SignOPT_inceptionv4_10000} \\\\
                    & SVM-OPT & {SVMOPT_inceptionv4_300} & {SVMOPT_inceptionv4_1000} & {SVMOPT_inceptionv4_2000} & {SVMOPT_inceptionv4_5000} & {SVMOPT_inceptionv4_8000} & {SVMOPT_inceptionv4_10000} & {targeted_SVMOPT_inceptionv4_300} & {targeted_SVMOPT_inceptionv4_1000} & {targeted_SVMOPT_inceptionv4_2000} & {targeted_SVMOPT_inceptionv4_5000} & {targeted_SVMOPT_inceptionv4_8000} & {targeted_SVMOPT_inceptionv4_10000} \\\\
                    & HopSkipJumpAttack & {HSJA_inceptionv4_300} & {HSJA_inceptionv4_1000} & {HSJA_inceptionv4_2000} & {HSJA_inceptionv4_5000} & {HSJA_inceptionv4_8000} & {HSJA_inceptionv4_10000} & {targeted_HSJA_inceptionv4_300} & {targeted_HSJA_inceptionv4_1000} & {targeted_HSJA_inceptionv4_2000} & {targeted_HSJA_inceptionv4_5000} & {targeted_HSJA_inceptionv4_8000} & {targeted_HSJA_inceptionv4_10000} \\\\
                    & Tangent Attack (ours) & {Tangent_inceptionv4_300} & {Tangent_inceptionv4_1000} & {Tangent_inceptionv4_2000} & {Tangent_inceptionv4_5000} & {Tangent_inceptionv4_8000} & {Tangent_inceptionv4_10000} & {targeted_Tangent_inceptionv4_300} & {targeted_Tangent_inceptionv4_1000} & {targeted_Tangent_inceptionv4_2000} & {targeted_Tangent_inceptionv4_5000} & {targeted_Tangent_inceptionv4_8000} & {targeted_Tangent_inceptionv4_10000} \\\\
                    \\midrule
                    & Boundary Attack & {BoundaryAttack_senet154_300} & {BoundaryAttack_senet154_1000} & {BoundaryAttack_senet154_2000} & {BoundaryAttack_senet154_5000} & {BoundaryAttack_senet154_8000} & {BoundaryAttack_senet154_10000} & {targeted_BoundaryAttack_senet154_300} & {targeted_BoundaryAttack_senet154_1000} & {targeted_BoundaryAttack_senet154_2000} & {targeted_BoundaryAttack_senet154_5000} & {targeted_BoundaryAttack_senet154_8000} & {targeted_BoundaryAttack_senet154_10000} \\\\
                    & Sign-OPT & {SignOPT_senet154_300} & {SignOPT_senet154_1000} & {SignOPT_senet154_2000} & {SignOPT_senet154_5000} & {SignOPT_senet154_8000} & {SignOPT_senet154_10000} & {targeted_SignOPT_senet154_300} & {targeted_SignOPT_senet154_1000} & {targeted_SignOPT_senet154_2000} & {targeted_SignOPT_senet154_5000} & {targeted_SignOPT_senet154_8000} & {targeted_SignOPT_senet154_10000} \\\\
                    & SVM-OPT & {SVMOPT_senet154_300} & {SVMOPT_senet154_1000} & {SVMOPT_senet154_2000} & {SVMOPT_senet154_5000} & {SVMOPT_senet154_8000} & {SVMOPT_senet154_10000} & {targeted_SVMOPT_senet154_300} & {targeted_SVMOPT_senet154_1000} & {targeted_SVMOPT_senet154_2000} & {targeted_SVMOPT_senet154_5000} & {targeted_SVMOPT_senet154_8000} & {targeted_SVMOPT_senet154_10000} \\\\
                    & HopSkipJumpAttack & {HSJA_senet154_300} & {HSJA_senet154_1000} & {HSJA_senet154_2000} & {HSJA_senet154_5000} & {HSJA_senet154_8000} & {HSJA_senet154_10000} & {targeted_HSJA_senet154_300} & {targeted_HSJA_senet154_1000} & {targeted_HSJA_senet154_2000} & {targeted_HSJA_senet154_5000} & {targeted_HSJA_senet154_8000} & {targeted_HSJA_senet154_10000} \\\\
                    & Tangent Attack (ours) & {Tangent_senet154_300} & {Tangent_senet154_1000} & {Tangent_senet154_2000} & {Tangent_senet154_5000} & {Tangent_senet154_8000} & {Tangent_senet154_10000} & {targeted_Tangent_senet154_300} & {targeted_Tangent_senet154_1000} & {targeted_Tangent_senet154_2000} & {targeted_Tangent_senet154_5000} & {targeted_Tangent_senet154_8000} & {targeted_Tangent_senet154_10000} \\\\
                    \\midrule
                    & Boundary Attack & {BoundaryAttack_resnet101_300} & {BoundaryAttack_resnet101_1000} & {BoundaryAttack_resnet101_2000} & {BoundaryAttack_resnet101_5000} & {BoundaryAttack_resnet101_8000} & {BoundaryAttack_resnet101_10000} & {targeted_BoundaryAttack_resnet101_300} & {targeted_BoundaryAttack_resnet101_1000} & {targeted_BoundaryAttack_resnet101_2000} & {targeted_BoundaryAttack_resnet101_5000} & {targeted_BoundaryAttack_resnet101_8000} & {targeted_BoundaryAttack_resnet101_10000} \\\\
                   & Sign-OPT & {SignOPT_resnet101_300} & {SignOPT_resnet101_1000} & {SignOPT_resnet101_2000} & {SignOPT_resnet101_5000} & {SignOPT_resnet101_8000} & {SignOPT_resnet101_10000} & {targeted_SignOPT_resnet101_300} & {targeted_SignOPT_resnet101_1000} & {targeted_SignOPT_resnet101_2000} & {targeted_SignOPT_resnet101_5000} & {targeted_SignOPT_resnet101_8000} & {targeted_SignOPT_resnet101_10000} \\\\
                    & SVM-OPT & {SVMOPT_resnet101_300} & {SVMOPT_resnet101_1000} & {SVMOPT_resnet101_2000} & {SVMOPT_resnet101_5000} & {SVMOPT_resnet101_8000} & {SVMOPT_resnet101_10000} & {targeted_SVMOPT_resnet101_300} & {targeted_SVMOPT_resnet101_1000} & {targeted_SVMOPT_resnet101_2000} & {targeted_SVMOPT_resnet101_5000} & {targeted_SVMOPT_resnet101_8000} & {targeted_SVMOPT_resnet101_10000} \\\\
                    & HopSkipJumpAttack & {HSJA_resnet101_300} & {HSJA_resnet101_1000} & {HSJA_resnet101_2000} & {HSJA_resnet101_5000} & {HSJA_resnet101_8000} & {HSJA_resnet101_10000} & {targeted_HSJA_resnet101_300} & {targeted_HSJA_resnet101_1000} & {targeted_HSJA_resnet101_2000} & {targeted_HSJA_resnet101_5000} & {targeted_HSJA_resnet101_8000} & {targeted_HSJA_resnet101_10000} \\\\
                    & Tangent Attack (ours) & {Tangent_resnet101_300} & {Tangent_resnet101_1000} & {Tangent_resnet101_2000} & {Tangent_resnet101_5000} & {Tangent_resnet101_8000} & {Tangent_resnet101_10000} & {targeted_Tangent_resnet101_300} & {targeted_Tangent_resnet101_1000} & {targeted_Tangent_resnet101_2000} & {targeted_Tangent_resnet101_5000} & {targeted_Tangent_resnet101_8000} & {targeted_Tangent_resnet101_10000} \\\\
                        """.format(

        SignOPT_inceptionv3_300=result["inceptionv3"]["Sign-OPT"][300],
        SignOPT_inceptionv3_1000=result["inceptionv3"]["Sign-OPT"][1000],
        SignOPT_inceptionv3_2000=result["inceptionv3"]["Sign-OPT"][2000],
        SignOPT_inceptionv3_5000=result["inceptionv3"]["Sign-OPT"][5000],
        SignOPT_inceptionv3_8000=result["inceptionv3"]["Sign-OPT"][8000],
        SignOPT_inceptionv3_10000=result["inceptionv3"]["Sign-OPT"][10000],

        SignOPT_inceptionv4_300=result["inceptionv4"]["Sign-OPT"][300],
        SignOPT_inceptionv4_1000=result["inceptionv4"]["Sign-OPT"][1000],
        SignOPT_inceptionv4_2000=result["inceptionv4"]["Sign-OPT"][2000],
        SignOPT_inceptionv4_5000=result["inceptionv4"]["Sign-OPT"][5000],
        SignOPT_inceptionv4_8000=result["inceptionv4"]["Sign-OPT"][8000],
        SignOPT_inceptionv4_10000=result["inceptionv4"]["Sign-OPT"][10000],

        SignOPT_senet154_300=result["senet154"]["Sign-OPT"][300],
        SignOPT_senet154_1000=result["senet154"]["Sign-OPT"][1000],
        SignOPT_senet154_2000=result["senet154"]["Sign-OPT"][2000],
        SignOPT_senet154_5000=result["senet154"]["Sign-OPT"][5000],
        SignOPT_senet154_8000=result["senet154"]["Sign-OPT"][8000],
        SignOPT_senet154_10000=result["senet154"]["Sign-OPT"][10000],

        SignOPT_resnet101_300=result["resnet101"]["Sign-OPT"][300],
        SignOPT_resnet101_1000=result["resnet101"]["Sign-OPT"][1000],
        SignOPT_resnet101_2000=result["resnet101"]["Sign-OPT"][2000],
        SignOPT_resnet101_5000=result["resnet101"]["Sign-OPT"][5000],
        SignOPT_resnet101_8000=result["resnet101"]["Sign-OPT"][8000],
        SignOPT_resnet101_10000=result["resnet101"]["Sign-OPT"][10000],


        SVMOPT_inceptionv3_300=result["inceptionv3"]["SVM-OPT"][300],
        SVMOPT_inceptionv3_1000=result["inceptionv3"]["SVM-OPT"][1000],
        SVMOPT_inceptionv3_2000=result["inceptionv3"]["SVM-OPT"][2000],
        SVMOPT_inceptionv3_5000=result["inceptionv3"]["SVM-OPT"][5000],
        SVMOPT_inceptionv3_8000=result["inceptionv3"]["SVM-OPT"][8000],
        SVMOPT_inceptionv3_10000=result["inceptionv3"]["SVM-OPT"][10000],

        SVMOPT_inceptionv4_300=result["inceptionv4"]["SVM-OPT"][300],
        SVMOPT_inceptionv4_1000=result["inceptionv4"]["SVM-OPT"][1000],
        SVMOPT_inceptionv4_2000=result["inceptionv4"]["SVM-OPT"][2000],
        SVMOPT_inceptionv4_5000=result["inceptionv4"]["SVM-OPT"][5000],
        SVMOPT_inceptionv4_8000=result["inceptionv4"]["SVM-OPT"][8000],
        SVMOPT_inceptionv4_10000=result["inceptionv4"]["SVM-OPT"][10000],

        SVMOPT_senet154_300=result["senet154"]["SVM-OPT"][300],
        SVMOPT_senet154_1000=result["senet154"]["SVM-OPT"][1000],
        SVMOPT_senet154_2000=result["senet154"]["SVM-OPT"][2000],
        SVMOPT_senet154_5000=result["senet154"]["SVM-OPT"][5000],
        SVMOPT_senet154_8000=result["senet154"]["SVM-OPT"][8000],
        SVMOPT_senet154_10000=result["senet154"]["SVM-OPT"][10000],

        SVMOPT_resnet101_300=result["resnet101"]["SVM-OPT"][300],
        SVMOPT_resnet101_1000=result["resnet101"]["SVM-OPT"][1000],
        SVMOPT_resnet101_2000=result["resnet101"]["SVM-OPT"][2000],
        SVMOPT_resnet101_5000=result["resnet101"]["SVM-OPT"][5000],
        SVMOPT_resnet101_8000=result["resnet101"]["SVM-OPT"][8000],
        SVMOPT_resnet101_10000=result["resnet101"]["SVM-OPT"][10000],


        BoundaryAttack_inceptionv3_300=result["inceptionv3"]["Boundary Attack"][300],
        BoundaryAttack_inceptionv3_1000=result["inceptionv3"]["Boundary Attack"][1000],
        BoundaryAttack_inceptionv3_2000=result["inceptionv3"]["Boundary Attack"][2000],
        BoundaryAttack_inceptionv3_5000=result["inceptionv3"]["Boundary Attack"][5000],
        BoundaryAttack_inceptionv3_8000=result["inceptionv3"]["Boundary Attack"][8000],
        BoundaryAttack_inceptionv3_10000=result["inceptionv3"]["Boundary Attack"][10000],

        BoundaryAttack_inceptionv4_300=result["inceptionv4"]["Boundary Attack"][300],
        BoundaryAttack_inceptionv4_1000=result["inceptionv4"]["Boundary Attack"][1000],
        BoundaryAttack_inceptionv4_2000=result["inceptionv4"]["Boundary Attack"][2000],
        BoundaryAttack_inceptionv4_5000=result["inceptionv4"]["Boundary Attack"][5000],
        BoundaryAttack_inceptionv4_8000=result["inceptionv4"]["Boundary Attack"][8000],
        BoundaryAttack_inceptionv4_10000=result["inceptionv4"]["Boundary Attack"][10000],

        BoundaryAttack_senet154_300=result["senet154"]["Boundary Attack"][300],
        BoundaryAttack_senet154_1000=result["senet154"]["Boundary Attack"][1000],
        BoundaryAttack_senet154_2000=result["senet154"]["Boundary Attack"][2000],
        BoundaryAttack_senet154_5000=result["senet154"]["Boundary Attack"][5000],
        BoundaryAttack_senet154_8000=result["senet154"]["Boundary Attack"][8000],
        BoundaryAttack_senet154_10000=result["senet154"]["Boundary Attack"][10000],

        BoundaryAttack_resnet101_300=result["resnet101"]["Boundary Attack"][300],
        BoundaryAttack_resnet101_1000=result["resnet101"]["Boundary Attack"][1000],
        BoundaryAttack_resnet101_2000=result["resnet101"]["Boundary Attack"][2000],
        BoundaryAttack_resnet101_5000=result["resnet101"]["Boundary Attack"][5000],
        BoundaryAttack_resnet101_8000=result["resnet101"]["Boundary Attack"][8000],
        BoundaryAttack_resnet101_10000=result["resnet101"]["Boundary Attack"][10000],


        HSJA_inceptionv3_300=result["inceptionv3"]["HopSkipJumpAttack"][300],
        HSJA_inceptionv3_1000=result["inceptionv3"]["HopSkipJumpAttack"][1000],
        HSJA_inceptionv3_2000=result["inceptionv3"]["HopSkipJumpAttack"][2000],
        HSJA_inceptionv3_5000=result["inceptionv3"]["HopSkipJumpAttack"][5000],
        HSJA_inceptionv3_8000=result["inceptionv3"]["HopSkipJumpAttack"][8000],
        HSJA_inceptionv3_10000=result["inceptionv3"]["HopSkipJumpAttack"][10000],

        HSJA_inceptionv4_300=result["inceptionv4"]["HopSkipJumpAttack"][300],
        HSJA_inceptionv4_1000=result["inceptionv4"]["HopSkipJumpAttack"][1000],
        HSJA_inceptionv4_2000=result["inceptionv4"]["HopSkipJumpAttack"][2000],
        HSJA_inceptionv4_5000=result["inceptionv4"]["HopSkipJumpAttack"][5000],
        HSJA_inceptionv4_8000=result["inceptionv4"]["HopSkipJumpAttack"][8000],
        HSJA_inceptionv4_10000=result["inceptionv4"]["HopSkipJumpAttack"][10000],

        HSJA_senet154_300=result["senet154"]["HopSkipJumpAttack"][300],
        HSJA_senet154_1000=result["senet154"]["HopSkipJumpAttack"][1000],
        HSJA_senet154_2000=result["senet154"]["HopSkipJumpAttack"][2000],
        HSJA_senet154_5000=result["senet154"]["HopSkipJumpAttack"][5000],
        HSJA_senet154_8000=result["senet154"]["HopSkipJumpAttack"][8000],
        HSJA_senet154_10000=result["senet154"]["HopSkipJumpAttack"][10000],

        HSJA_resnet101_300=result["resnet101"]["HopSkipJumpAttack"][300],
        HSJA_resnet101_1000=result["resnet101"]["HopSkipJumpAttack"][1000],
        HSJA_resnet101_2000=result["resnet101"]["HopSkipJumpAttack"][2000],
        HSJA_resnet101_5000=result["resnet101"]["HopSkipJumpAttack"][5000],
        HSJA_resnet101_8000=result["resnet101"]["HopSkipJumpAttack"][8000],
        HSJA_resnet101_10000=result["resnet101"]["HopSkipJumpAttack"][10000],



        Tangent_inceptionv3_300=result["inceptionv3"]["Tangent Attack"][300],
        Tangent_inceptionv3_1000=result["inceptionv3"]["Tangent Attack"][1000],
        Tangent_inceptionv3_2000=result["inceptionv3"]["Tangent Attack"][2000],
        Tangent_inceptionv3_5000=result["inceptionv3"]["Tangent Attack"][5000],
        Tangent_inceptionv3_8000=result["inceptionv3"]["Tangent Attack"][8000],
        Tangent_inceptionv3_10000=result["inceptionv3"]["Tangent Attack"][10000],

        Tangent_inceptionv4_300=result["inceptionv4"]["Tangent Attack"][300],
        Tangent_inceptionv4_1000=result["inceptionv4"]["Tangent Attack"][1000],
        Tangent_inceptionv4_2000=result["inceptionv4"]["Tangent Attack"][2000],
        Tangent_inceptionv4_5000=result["inceptionv4"]["Tangent Attack"][5000],
        Tangent_inceptionv4_8000=result["inceptionv4"]["Tangent Attack"][8000],
        Tangent_inceptionv4_10000=result["inceptionv4"]["Tangent Attack"][10000],

        Tangent_senet154_300=result["senet154"]["Tangent Attack"][300],
        Tangent_senet154_1000=result["senet154"]["Tangent Attack"][1000],
        Tangent_senet154_2000=result["senet154"]["Tangent Attack"][2000],
        Tangent_senet154_5000=result["senet154"]["Tangent Attack"][5000],
        Tangent_senet154_8000=result["senet154"]["Tangent Attack"][8000],
        Tangent_senet154_10000=result["senet154"]["Tangent Attack"][10000],

        Tangent_resnet101_300=result["resnet101"]["Tangent Attack"][300],
        Tangent_resnet101_1000=result["resnet101"]["Tangent Attack"][1000],
        Tangent_resnet101_2000=result["resnet101"]["Tangent Attack"][2000],
        Tangent_resnet101_5000=result["resnet101"]["Tangent Attack"][5000],
        Tangent_resnet101_8000=result["resnet101"]["Tangent Attack"][8000],
        Tangent_resnet101_10000=result["resnet101"]["Tangent Attack"][10000],



        targeted_SignOPT_inceptionv3_300=targeted_attack_result["inceptionv3"]["Sign-OPT"][300],
        targeted_SignOPT_inceptionv3_1000=targeted_attack_result["inceptionv3"]["Sign-OPT"][1000],
        targeted_SignOPT_inceptionv3_2000=targeted_attack_result["inceptionv3"]["Sign-OPT"][2000],
        targeted_SignOPT_inceptionv3_5000=targeted_attack_result["inceptionv3"]["Sign-OPT"][5000],
        targeted_SignOPT_inceptionv3_8000=targeted_attack_result["inceptionv3"]["Sign-OPT"][8000],
        targeted_SignOPT_inceptionv3_10000=targeted_attack_result["inceptionv3"]["Sign-OPT"][10000],
        targeted_SignOPT_inceptionv4_300=targeted_attack_result["inceptionv4"]["Sign-OPT"][300],
        targeted_SignOPT_inceptionv4_1000=targeted_attack_result["inceptionv4"]["Sign-OPT"][1000],
        targeted_SignOPT_inceptionv4_2000=targeted_attack_result["inceptionv4"]["Sign-OPT"][2000],
        targeted_SignOPT_inceptionv4_5000=targeted_attack_result["inceptionv4"]["Sign-OPT"][5000],
        targeted_SignOPT_inceptionv4_8000=targeted_attack_result["inceptionv4"]["Sign-OPT"][8000],
        targeted_SignOPT_inceptionv4_10000=targeted_attack_result["inceptionv4"]["Sign-OPT"][10000],
        targeted_SignOPT_senet154_300=targeted_attack_result["senet154"]["Sign-OPT"][300],
        targeted_SignOPT_senet154_1000=targeted_attack_result["senet154"]["Sign-OPT"][1000],
        targeted_SignOPT_senet154_2000=targeted_attack_result["senet154"]["Sign-OPT"][2000],
        targeted_SignOPT_senet154_5000=targeted_attack_result["senet154"]["Sign-OPT"][5000],
        targeted_SignOPT_senet154_8000=targeted_attack_result["senet154"]["Sign-OPT"][8000],
        targeted_SignOPT_senet154_10000=targeted_attack_result["senet154"]["Sign-OPT"][10000],
        targeted_SignOPT_resnet101_300=targeted_attack_result["resnet101"]["Sign-OPT"][300],
        targeted_SignOPT_resnet101_1000=targeted_attack_result["resnet101"]["Sign-OPT"][1000],
        targeted_SignOPT_resnet101_2000=targeted_attack_result["resnet101"]["Sign-OPT"][2000],
        targeted_SignOPT_resnet101_5000=targeted_attack_result["resnet101"]["Sign-OPT"][5000],
        targeted_SignOPT_resnet101_8000=targeted_attack_result["resnet101"]["Sign-OPT"][8000],
        targeted_SignOPT_resnet101_10000=targeted_attack_result["resnet101"]["Sign-OPT"][10000],


        targeted_SVMOPT_inceptionv3_300=targeted_attack_result["inceptionv3"]["SVM-OPT"][300],
        targeted_SVMOPT_inceptionv3_1000=targeted_attack_result["inceptionv3"]["SVM-OPT"][1000],
        targeted_SVMOPT_inceptionv3_2000=targeted_attack_result["inceptionv3"]["SVM-OPT"][2000],
        targeted_SVMOPT_inceptionv3_5000=targeted_attack_result["inceptionv3"]["SVM-OPT"][5000],
        targeted_SVMOPT_inceptionv3_8000=targeted_attack_result["inceptionv3"]["SVM-OPT"][8000],
        targeted_SVMOPT_inceptionv3_10000=targeted_attack_result["inceptionv3"]["SVM-OPT"][10000],

        targeted_SVMOPT_inceptionv4_300=targeted_attack_result["inceptionv4"]["SVM-OPT"][300],
        targeted_SVMOPT_inceptionv4_1000=targeted_attack_result["inceptionv4"]["SVM-OPT"][1000],
        targeted_SVMOPT_inceptionv4_2000=targeted_attack_result["inceptionv4"]["SVM-OPT"][2000],
        targeted_SVMOPT_inceptionv4_5000=targeted_attack_result["inceptionv4"]["SVM-OPT"][5000],
        targeted_SVMOPT_inceptionv4_8000=targeted_attack_result["inceptionv4"]["SVM-OPT"][8000],
        targeted_SVMOPT_inceptionv4_10000=targeted_attack_result["inceptionv4"]["SVM-OPT"][10000],

        targeted_SVMOPT_senet154_300=targeted_attack_result["senet154"]["SVM-OPT"][300],
        targeted_SVMOPT_senet154_1000=targeted_attack_result["senet154"]["SVM-OPT"][1000],
        targeted_SVMOPT_senet154_2000=targeted_attack_result["senet154"]["SVM-OPT"][2000],
        targeted_SVMOPT_senet154_5000=targeted_attack_result["senet154"]["SVM-OPT"][5000],
        targeted_SVMOPT_senet154_8000=targeted_attack_result["senet154"]["SVM-OPT"][8000],
        targeted_SVMOPT_senet154_10000=targeted_attack_result["senet154"]["SVM-OPT"][10000],

        targeted_SVMOPT_resnet101_300=targeted_attack_result["resnet101"]["SVM-OPT"][300],
        targeted_SVMOPT_resnet101_1000=targeted_attack_result["resnet101"]["SVM-OPT"][1000],
        targeted_SVMOPT_resnet101_2000=targeted_attack_result["resnet101"]["SVM-OPT"][2000],
        targeted_SVMOPT_resnet101_5000=targeted_attack_result["resnet101"]["SVM-OPT"][5000],
        targeted_SVMOPT_resnet101_8000=targeted_attack_result["resnet101"]["SVM-OPT"][8000],
        targeted_SVMOPT_resnet101_10000=targeted_attack_result["resnet101"]["SVM-OPT"][10000],


        targeted_BoundaryAttack_inceptionv3_300=targeted_attack_result["inceptionv3"]["Boundary Attack"][300],
        targeted_BoundaryAttack_inceptionv3_1000=targeted_attack_result["inceptionv3"]["Boundary Attack"][1000],
        targeted_BoundaryAttack_inceptionv3_2000=targeted_attack_result["inceptionv3"]["Boundary Attack"][2000],
        targeted_BoundaryAttack_inceptionv3_5000=targeted_attack_result["inceptionv3"]["Boundary Attack"][5000],
        targeted_BoundaryAttack_inceptionv3_8000=targeted_attack_result["inceptionv3"]["Boundary Attack"][8000],
        targeted_BoundaryAttack_inceptionv3_10000=targeted_attack_result["inceptionv3"]["Boundary Attack"][10000],

        targeted_BoundaryAttack_inceptionv4_300=targeted_attack_result["inceptionv4"]["Boundary Attack"][300],
        targeted_BoundaryAttack_inceptionv4_1000=targeted_attack_result["inceptionv4"]["Boundary Attack"][1000],
        targeted_BoundaryAttack_inceptionv4_2000=targeted_attack_result["inceptionv4"]["Boundary Attack"][2000],
        targeted_BoundaryAttack_inceptionv4_5000=targeted_attack_result["inceptionv4"]["Boundary Attack"][5000],
        targeted_BoundaryAttack_inceptionv4_8000=targeted_attack_result["inceptionv4"]["Boundary Attack"][8000],
        targeted_BoundaryAttack_inceptionv4_10000=targeted_attack_result["inceptionv4"]["Boundary Attack"][10000],

        targeted_BoundaryAttack_senet154_300=targeted_attack_result["senet154"]["Boundary Attack"][300],
        targeted_BoundaryAttack_senet154_1000=targeted_attack_result["senet154"]["Boundary Attack"][1000],
        targeted_BoundaryAttack_senet154_2000=targeted_attack_result["senet154"]["Boundary Attack"][2000],
        targeted_BoundaryAttack_senet154_5000=targeted_attack_result["senet154"]["Boundary Attack"][5000],
        targeted_BoundaryAttack_senet154_8000=targeted_attack_result["senet154"]["Boundary Attack"][8000],
        targeted_BoundaryAttack_senet154_10000=targeted_attack_result["senet154"]["Boundary Attack"][10000],

        targeted_BoundaryAttack_resnet101_300=targeted_attack_result["resnet101"]["Boundary Attack"][300],
        targeted_BoundaryAttack_resnet101_1000=targeted_attack_result["resnet101"]["Boundary Attack"][1000],
        targeted_BoundaryAttack_resnet101_2000=targeted_attack_result["resnet101"]["Boundary Attack"][2000],
        targeted_BoundaryAttack_resnet101_5000=targeted_attack_result["resnet101"]["Boundary Attack"][5000],
        targeted_BoundaryAttack_resnet101_8000=targeted_attack_result["resnet101"]["Boundary Attack"][8000],
        targeted_BoundaryAttack_resnet101_10000=targeted_attack_result["resnet101"]["Boundary Attack"][10000],


        targeted_HSJA_inceptionv3_300=targeted_attack_result["inceptionv3"]["HopSkipJumpAttack"][300],
        targeted_HSJA_inceptionv3_1000=targeted_attack_result["inceptionv3"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_inceptionv3_2000=targeted_attack_result["inceptionv3"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_inceptionv3_5000=targeted_attack_result["inceptionv3"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_inceptionv3_8000=targeted_attack_result["inceptionv3"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_inceptionv3_10000=targeted_attack_result["inceptionv3"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_inceptionv4_300=targeted_attack_result["inceptionv4"]["HopSkipJumpAttack"][300],
        targeted_HSJA_inceptionv4_1000=targeted_attack_result["inceptionv4"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_inceptionv4_2000=targeted_attack_result["inceptionv4"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_inceptionv4_5000=targeted_attack_result["inceptionv4"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_inceptionv4_8000=targeted_attack_result["inceptionv4"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_inceptionv4_10000=targeted_attack_result["inceptionv4"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_senet154_300=targeted_attack_result["senet154"]["HopSkipJumpAttack"][300],
        targeted_HSJA_senet154_1000=targeted_attack_result["senet154"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_senet154_2000=targeted_attack_result["senet154"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_senet154_5000=targeted_attack_result["senet154"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_senet154_8000=targeted_attack_result["senet154"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_senet154_10000=targeted_attack_result["senet154"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_resnet101_300=targeted_attack_result["resnet101"]["HopSkipJumpAttack"][300],
        targeted_HSJA_resnet101_1000=targeted_attack_result["resnet101"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_resnet101_2000=targeted_attack_result["resnet101"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_resnet101_5000=targeted_attack_result["resnet101"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_resnet101_8000=targeted_attack_result["resnet101"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_resnet101_10000=targeted_attack_result["resnet101"]["HopSkipJumpAttack"][10000],


        targeted_Tangent_inceptionv3_300=targeted_attack_result["inceptionv3"]["Tangent Attack"][300],
        targeted_Tangent_inceptionv3_1000=targeted_attack_result["inceptionv3"]["Tangent Attack"][1000],
        targeted_Tangent_inceptionv3_2000=targeted_attack_result["inceptionv3"]["Tangent Attack"][2000],
        targeted_Tangent_inceptionv3_5000=targeted_attack_result["inceptionv3"]["Tangent Attack"][5000],
        targeted_Tangent_inceptionv3_8000=targeted_attack_result["inceptionv3"]["Tangent Attack"][8000],
        targeted_Tangent_inceptionv3_10000=targeted_attack_result["inceptionv3"]["Tangent Attack"][10000],

        targeted_Tangent_inceptionv4_300=targeted_attack_result["inceptionv4"]["Tangent Attack"][300],
        targeted_Tangent_inceptionv4_1000=targeted_attack_result["inceptionv4"]["Tangent Attack"][1000],
        targeted_Tangent_inceptionv4_2000=targeted_attack_result["inceptionv4"]["Tangent Attack"][2000],
        targeted_Tangent_inceptionv4_5000=targeted_attack_result["inceptionv4"]["Tangent Attack"][5000],
        targeted_Tangent_inceptionv4_8000=targeted_attack_result["inceptionv4"]["Tangent Attack"][8000],
        targeted_Tangent_inceptionv4_10000=targeted_attack_result["inceptionv4"]["Tangent Attack"][10000],

        targeted_Tangent_senet154_300=targeted_attack_result["senet154"]["Tangent Attack"][300],
        targeted_Tangent_senet154_1000=targeted_attack_result["senet154"]["Tangent Attack"][1000],
        targeted_Tangent_senet154_2000=targeted_attack_result["senet154"]["Tangent Attack"][2000],
        targeted_Tangent_senet154_5000=targeted_attack_result["senet154"]["Tangent Attack"][5000],
        targeted_Tangent_senet154_8000=targeted_attack_result["senet154"]["Tangent Attack"][8000],
        targeted_Tangent_senet154_10000=targeted_attack_result["senet154"]["Tangent Attack"][10000],

        targeted_Tangent_resnet101_300=targeted_attack_result["resnet101"]["Tangent Attack"][300],
        targeted_Tangent_resnet101_1000=targeted_attack_result["resnet101"]["Tangent Attack"][1000],
        targeted_Tangent_resnet101_2000=targeted_attack_result["resnet101"]["Tangent Attack"][2000],
        targeted_Tangent_resnet101_5000=targeted_attack_result["resnet101"]["Tangent Attack"][5000],
        targeted_Tangent_resnet101_8000=targeted_attack_result["resnet101"]["Tangent Attack"][8000],
        targeted_Tangent_resnet101_10000=targeted_attack_result["resnet101"]["Tangent Attack"][10000],


    )
    )

def draw_wide_table_CIFAR(untargeted_result, targeted_result):

    print("""
                    & GeoDA & {untargeted_GeoDA_pyramidnet272_1000} & {untargeted_GeoDA_pyramidnet272_2000} & {untargeted_GeoDA_pyramidnet272_5000} & {untargeted_GeoDA_pyramidnet272_8000} & {untargeted_GeoDA_pyramidnet272_10000} & {targeted_GeoDA_pyramidnet272_1000} & {targeted_GeoDA_pyramidnet272_2000} & {targeted_GeoDA_pyramidnet272_5000} & {targeted_GeoDA_pyramidnet272_8000} & {targeted_GeoDA_pyramidnet272_10000}  \\\\
                    & RayS & {untargeted_RayS_pyramidnet272_1000} & {untargeted_RayS_pyramidnet272_2000} & {untargeted_RayS_pyramidnet272_5000} & {untargeted_RayS_pyramidnet272_8000} & {untargeted_RayS_pyramidnet272_10000} & {targeted_RayS_pyramidnet272_1000} & {targeted_RayS_pyramidnet272_2000} & {targeted_RayS_pyramidnet272_5000} & {targeted_RayS_pyramidnet272_8000} & {targeted_RayS_pyramidnet272_10000}  \\\\
                    & Sign-OPT & {untargeted_SignOPT_pyramidnet272_1000} & {untargeted_SignOPT_pyramidnet272_2000} & {untargeted_SignOPT_pyramidnet272_5000} & {untargeted_SignOPT_pyramidnet272_8000} & {untargeted_SignOPT_pyramidnet272_10000} & {targeted_SignOPT_pyramidnet272_1000} & {targeted_SignOPT_pyramidnet272_2000} & {targeted_SignOPT_pyramidnet272_5000} & {targeted_SignOPT_pyramidnet272_8000} & {targeted_SignOPT_pyramidnet272_10000}  \\\\
                    & SVM-OPT & {untargeted_SVMOPT_pyramidnet272_1000} & {untargeted_SVMOPT_pyramidnet272_2000} & {untargeted_SVMOPT_pyramidnet272_5000} & {untargeted_SVMOPT_pyramidnet272_8000} & {untargeted_SVMOPT_pyramidnet272_10000} & {targeted_SVMOPT_pyramidnet272_1000} & {targeted_SVMOPT_pyramidnet272_2000} & {targeted_SVMOPT_pyramidnet272_5000} & {targeted_SVMOPT_pyramidnet272_8000} & {targeted_SVMOPT_pyramidnet272_10000}  \\\\
                    & HopSkipJumpAttack & {untargeted_HSJA_pyramidnet272_1000} & {untargeted_HSJA_pyramidnet272_2000} & {untargeted_HSJA_pyramidnet272_5000} & {untargeted_HSJA_pyramidnet272_8000} & {untargeted_HSJA_pyramidnet272_10000} & {targeted_HSJA_pyramidnet272_1000} & {targeted_HSJA_pyramidnet272_2000} & {targeted_HSJA_pyramidnet272_5000} & {targeted_HSJA_pyramidnet272_8000} & {targeted_HSJA_pyramidnet272_10000}  \\\\
                    & Tangent Attack (ours) & {untargeted_Tangent_pyramidnet272_1000} & {untargeted_Tangent_pyramidnet272_2000} & {untargeted_Tangent_pyramidnet272_5000} & {untargeted_Tangent_pyramidnet272_8000} & {untargeted_Tangent_pyramidnet272_10000} & {targeted_Tangent_pyramidnet272_1000} & {targeted_Tangent_pyramidnet272_2000} & {targeted_Tangent_pyramidnet272_5000} & {targeted_Tangent_pyramidnet272_8000} & {targeted_Tangent_pyramidnet272_10000}  \\\\
                    \\midrule
                    & GeoDA & {untargeted_GeoDA_gdas_1000} & {untargeted_GeoDA_gdas_2000} & {untargeted_GeoDA_gdas_5000} & {untargeted_GeoDA_gdas_8000} & {untargeted_GeoDA_gdas_10000} & {targeted_GeoDA_gdas_1000} & {targeted_GeoDA_gdas_2000} & {targeted_GeoDA_gdas_5000} & {targeted_GeoDA_gdas_8000} & {targeted_GeoDA_gdas_10000}  \\\\
                    & RayS & {untargeted_RayS_gdas_1000} & {untargeted_RayS_gdas_2000} & {untargeted_RayS_gdas_5000} & {untargeted_RayS_gdas_8000} & {untargeted_RayS_gdas_10000} & {targeted_RayS_gdas_1000} & {targeted_RayS_gdas_2000} & {targeted_RayS_gdas_5000} & {targeted_RayS_gdas_8000} & {targeted_RayS_gdas_10000}  \\\\
                    & Sign-OPT & {untargeted_SignOPT_gdas_1000} & {untargeted_SignOPT_gdas_2000} & {untargeted_SignOPT_gdas_5000} & {untargeted_SignOPT_gdas_8000} & {untargeted_SignOPT_gdas_10000} & {targeted_SignOPT_gdas_1000} & {targeted_SignOPT_gdas_2000} & {targeted_SignOPT_gdas_5000} & {targeted_SignOPT_gdas_8000} & {targeted_SignOPT_gdas_10000}  \\\\
                    & SVM-OPT & {untargeted_SVMOPT_gdas_1000} & {untargeted_SVMOPT_gdas_2000} & {untargeted_SVMOPT_gdas_5000} & {untargeted_SVMOPT_gdas_8000} & {untargeted_SVMOPT_gdas_10000} & {targeted_SVMOPT_gdas_1000} & {targeted_SVMOPT_gdas_2000} & {targeted_SVMOPT_gdas_5000} & {targeted_SVMOPT_gdas_8000} & {targeted_SVMOPT_gdas_10000}  \\\\
                    & HopSkipJumpAttack & {untargeted_HSJA_gdas_1000} & {untargeted_HSJA_gdas_2000} & {untargeted_HSJA_gdas_5000} & {untargeted_HSJA_gdas_8000} & {untargeted_HSJA_gdas_10000} & {targeted_HSJA_gdas_1000} & {targeted_HSJA_gdas_2000} & {targeted_HSJA_gdas_5000} & {targeted_HSJA_gdas_8000} & {targeted_HSJA_gdas_10000}  \\\\
                    & Tangent Attack (ours) & {untargeted_Tangent_gdas_1000} & {untargeted_Tangent_gdas_2000} & {untargeted_Tangent_gdas_5000} & {untargeted_Tangent_gdas_8000} & {untargeted_Tangent_gdas_10000} & {targeted_Tangent_gdas_1000} & {targeted_Tangent_gdas_2000} & {targeted_Tangent_gdas_5000} & {targeted_Tangent_gdas_8000} & {targeted_Tangent_gdas_10000}  \\\\
                    \\midrule
                    & GeoDA & {untargeted_GeoDA_WRN28_1000} & {untargeted_GeoDA_WRN28_2000} & {untargeted_GeoDA_WRN28_5000} & {untargeted_GeoDA_WRN28_8000} & {untargeted_GeoDA_WRN28_10000} & {targeted_GeoDA_WRN28_1000} & {targeted_GeoDA_WRN28_2000} & {targeted_GeoDA_WRN28_5000} & {targeted_GeoDA_WRN28_8000} & {targeted_GeoDA_WRN28_10000}  \\\\
                    & RayS & {untargeted_RayS_WRN28_1000} & {untargeted_RayS_WRN28_2000} & {untargeted_RayS_WRN28_5000} & {untargeted_RayS_WRN28_8000} & {untargeted_RayS_WRN28_10000} & {targeted_RayS_WRN28_1000} & {targeted_RayS_WRN28_2000} & {targeted_RayS_WRN28_5000} & {targeted_RayS_WRN28_8000} & {targeted_RayS_WRN28_10000}  \\\\
                    & Sign-OPT & {untargeted_SignOPT_WRN28_1000} & {untargeted_SignOPT_WRN28_2000} & {untargeted_SignOPT_WRN28_5000} & {untargeted_SignOPT_WRN28_8000} & {untargeted_SignOPT_WRN28_10000} & {targeted_SignOPT_WRN28_1000} & {targeted_SignOPT_WRN28_2000} & {targeted_SignOPT_WRN28_5000} & {targeted_SignOPT_WRN28_8000} & {targeted_SignOPT_WRN28_10000}  \\\\
                    & SVM-OPT & {untargeted_SVMOPT_WRN28_1000} & {untargeted_SVMOPT_WRN28_2000} & {untargeted_SVMOPT_WRN28_5000} & {untargeted_SVMOPT_WRN28_8000} & {untargeted_SVMOPT_WRN28_10000} & {targeted_SVMOPT_WRN28_1000} & {targeted_SVMOPT_WRN28_2000} & {targeted_SVMOPT_WRN28_5000} & {targeted_SVMOPT_WRN28_8000} & {targeted_SVMOPT_WRN28_10000}  \\\\
                    & HopSkipJumpAttack & {untargeted_HSJA_WRN28_1000} & {untargeted_HSJA_WRN28_2000} & {untargeted_HSJA_WRN28_5000} & {untargeted_HSJA_WRN28_8000} & {untargeted_HSJA_WRN28_10000} & {targeted_HSJA_WRN28_1000} & {targeted_HSJA_WRN28_2000} & {targeted_HSJA_WRN28_5000} & {targeted_HSJA_WRN28_8000} & {targeted_HSJA_WRN28_10000}  \\\\
                    & Tangent Attack (ours) & {untargeted_Tangent_WRN28_1000} & {untargeted_Tangent_WRN28_2000} & {untargeted_Tangent_WRN28_5000} & {untargeted_Tangent_WRN28_8000} & {untargeted_Tangent_WRN28_10000} & {targeted_Tangent_WRN28_1000} & {targeted_Tangent_WRN28_2000} & {targeted_Tangent_WRN28_5000} & {targeted_Tangent_WRN28_8000} & {targeted_Tangent_WRN28_10000}  \\\\
                    \\midrule
                    & GeoDA & {untargeted_GeoDA_WRN40_1000} & {untargeted_GeoDA_WRN40_2000} & {untargeted_GeoDA_WRN40_5000} & {untargeted_GeoDA_WRN40_8000} & {untargeted_GeoDA_WRN40_10000} & {targeted_GeoDA_WRN40_1000} & {targeted_GeoDA_WRN40_2000} & {targeted_GeoDA_WRN40_5000} & {targeted_GeoDA_WRN40_8000} & {targeted_GeoDA_WRN40_10000}  \\\\
                    & RayS & {untargeted_RayS_WRN40_1000} & {untargeted_RayS_WRN40_2000} & {untargeted_RayS_WRN40_5000} & {untargeted_RayS_WRN40_8000} & {untargeted_RayS_WRN40_10000} & {targeted_RayS_WRN40_1000} & {targeted_RayS_WRN40_2000} & {targeted_RayS_WRN40_5000} & {targeted_RayS_WRN40_8000} & {targeted_RayS_WRN40_10000}  \\\\
                    & Sign-OPT & {untargeted_SignOPT_WRN40_1000} & {untargeted_SignOPT_WRN40_2000} & {untargeted_SignOPT_WRN40_5000} & {untargeted_SignOPT_WRN40_8000} & {untargeted_SignOPT_WRN40_10000} & {targeted_SignOPT_WRN40_1000} & {targeted_SignOPT_WRN40_2000} & {targeted_SignOPT_WRN40_5000} & {targeted_SignOPT_WRN40_8000} & {targeted_SignOPT_WRN40_10000}  \\\\
                    & SVM-OPT & {untargeted_SVMOPT_WRN40_1000} & {untargeted_SVMOPT_WRN40_2000} & {untargeted_SVMOPT_WRN40_5000} & {untargeted_SVMOPT_WRN40_8000} & {untargeted_SVMOPT_WRN40_10000} & {targeted_SVMOPT_WRN40_1000} & {targeted_SVMOPT_WRN40_2000} & {targeted_SVMOPT_WRN40_5000} & {targeted_SVMOPT_WRN40_8000} & {targeted_SVMOPT_WRN40_10000}  \\\\
                    & HopSkipJumpAttack & {untargeted_HSJA_WRN40_1000} & {untargeted_HSJA_WRN40_2000} & {untargeted_HSJA_WRN40_5000} & {untargeted_HSJA_WRN40_8000} & {untargeted_HSJA_WRN40_10000} & {targeted_HSJA_WRN40_1000} & {targeted_HSJA_WRN40_2000} & {targeted_HSJA_WRN40_5000} & {targeted_HSJA_WRN40_8000} & {targeted_HSJA_WRN40_10000}  \\\\
                    & Tangent Attack (ours) & {untargeted_Tangent_WRN40_1000} & {untargeted_Tangent_WRN40_2000} & {untargeted_Tangent_WRN40_5000} & {untargeted_Tangent_WRN40_8000} & {untargeted_Tangent_WRN40_10000} & {targeted_Tangent_WRN40_1000} & {targeted_Tangent_WRN40_2000} & {targeted_Tangent_WRN40_5000} & {targeted_Tangent_WRN40_8000} & {targeted_Tangent_WRN40_10000}  \\\\
                        """.format(

        untargeted_SignOPT_pyramidnet272_1000=untargeted_result["pyramidnet272"]["Sign-OPT"][1000],
        untargeted_SignOPT_pyramidnet272_2000=untargeted_result["pyramidnet272"]["Sign-OPT"][2000],
        untargeted_SignOPT_pyramidnet272_5000=untargeted_result["pyramidnet272"]["Sign-OPT"][5000],
        untargeted_SignOPT_pyramidnet272_8000=untargeted_result["pyramidnet272"]["Sign-OPT"][8000],
        untargeted_SignOPT_pyramidnet272_10000=untargeted_result["pyramidnet272"]["Sign-OPT"][10000],

        targeted_SignOPT_pyramidnet272_1000=targeted_result["pyramidnet272"]["Sign-OPT"][1000],
        targeted_SignOPT_pyramidnet272_2000=targeted_result["pyramidnet272"]["Sign-OPT"][2000],
        targeted_SignOPT_pyramidnet272_5000=targeted_result["pyramidnet272"]["Sign-OPT"][5000],
        targeted_SignOPT_pyramidnet272_8000=targeted_result["pyramidnet272"]["Sign-OPT"][8000],
        targeted_SignOPT_pyramidnet272_10000=targeted_result["pyramidnet272"]["Sign-OPT"][10000],

        untargeted_SignOPT_gdas_1000=untargeted_result["gdas"]["Sign-OPT"][1000],
        untargeted_SignOPT_gdas_2000=untargeted_result["gdas"]["Sign-OPT"][2000],
        untargeted_SignOPT_gdas_5000=untargeted_result["gdas"]["Sign-OPT"][5000],
        untargeted_SignOPT_gdas_8000=untargeted_result["gdas"]["Sign-OPT"][8000],
        untargeted_SignOPT_gdas_10000=untargeted_result["gdas"]["Sign-OPT"][10000],

        targeted_SignOPT_gdas_1000=targeted_result["gdas"]["Sign-OPT"][1000],
        targeted_SignOPT_gdas_2000=targeted_result["gdas"]["Sign-OPT"][2000],
        targeted_SignOPT_gdas_5000=targeted_result["gdas"]["Sign-OPT"][5000],
        targeted_SignOPT_gdas_8000=targeted_result["gdas"]["Sign-OPT"][8000],
        targeted_SignOPT_gdas_10000=targeted_result["gdas"]["Sign-OPT"][10000],

        untargeted_SignOPT_WRN28_1000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][1000],
        untargeted_SignOPT_WRN28_2000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][2000],
        untargeted_SignOPT_WRN28_5000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][5000],
        untargeted_SignOPT_WRN28_8000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][8000],
        untargeted_SignOPT_WRN28_10000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][10000],

        targeted_SignOPT_WRN28_1000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][1000],
        targeted_SignOPT_WRN28_2000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][2000],
        targeted_SignOPT_WRN28_5000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][5000],
        targeted_SignOPT_WRN28_8000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][8000],
        targeted_SignOPT_WRN28_10000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][10000],

        untargeted_SignOPT_WRN40_1000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][1000],
        untargeted_SignOPT_WRN40_2000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][2000],
        untargeted_SignOPT_WRN40_5000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][5000],
        untargeted_SignOPT_WRN40_8000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][8000],
        untargeted_SignOPT_WRN40_10000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][10000],

        targeted_SignOPT_WRN40_1000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][1000],
        targeted_SignOPT_WRN40_2000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][2000],
        targeted_SignOPT_WRN40_5000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][5000],
        targeted_SignOPT_WRN40_8000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][8000],
        targeted_SignOPT_WRN40_10000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][10000],

        untargeted_SVMOPT_pyramidnet272_1000=untargeted_result["pyramidnet272"]["SVM-OPT"][1000],
        untargeted_SVMOPT_pyramidnet272_2000=untargeted_result["pyramidnet272"]["SVM-OPT"][2000],
        untargeted_SVMOPT_pyramidnet272_5000=untargeted_result["pyramidnet272"]["SVM-OPT"][5000],
        untargeted_SVMOPT_pyramidnet272_8000=untargeted_result["pyramidnet272"]["SVM-OPT"][8000],
        untargeted_SVMOPT_pyramidnet272_10000=untargeted_result["pyramidnet272"]["SVM-OPT"][10000],

        targeted_SVMOPT_pyramidnet272_1000=targeted_result["pyramidnet272"]["SVM-OPT"][1000],
        targeted_SVMOPT_pyramidnet272_2000=targeted_result["pyramidnet272"]["SVM-OPT"][2000],
        targeted_SVMOPT_pyramidnet272_5000=targeted_result["pyramidnet272"]["SVM-OPT"][5000],
        targeted_SVMOPT_pyramidnet272_8000=targeted_result["pyramidnet272"]["SVM-OPT"][8000],
        targeted_SVMOPT_pyramidnet272_10000=targeted_result["pyramidnet272"]["SVM-OPT"][10000],

        untargeted_SVMOPT_gdas_1000=untargeted_result["gdas"]["SVM-OPT"][1000],
        untargeted_SVMOPT_gdas_2000=untargeted_result["gdas"]["SVM-OPT"][2000],
        untargeted_SVMOPT_gdas_5000=untargeted_result["gdas"]["SVM-OPT"][5000],
        untargeted_SVMOPT_gdas_8000=untargeted_result["gdas"]["SVM-OPT"][8000],
        untargeted_SVMOPT_gdas_10000=untargeted_result["gdas"]["SVM-OPT"][10000],

        targeted_SVMOPT_gdas_1000=targeted_result["gdas"]["SVM-OPT"][1000],
        targeted_SVMOPT_gdas_2000=targeted_result["gdas"]["SVM-OPT"][2000],
        targeted_SVMOPT_gdas_5000=targeted_result["gdas"]["SVM-OPT"][5000],
        targeted_SVMOPT_gdas_8000=targeted_result["gdas"]["SVM-OPT"][8000],
        targeted_SVMOPT_gdas_10000=targeted_result["gdas"]["SVM-OPT"][10000],

        untargeted_SVMOPT_WRN28_1000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][1000],
        untargeted_SVMOPT_WRN28_2000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][2000],
        untargeted_SVMOPT_WRN28_5000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][5000],
        untargeted_SVMOPT_WRN28_8000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][8000],
        untargeted_SVMOPT_WRN28_10000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][10000],

        targeted_SVMOPT_WRN28_1000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][1000],
        targeted_SVMOPT_WRN28_2000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][2000],
        targeted_SVMOPT_WRN28_5000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][5000],
        targeted_SVMOPT_WRN28_8000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][8000],
        targeted_SVMOPT_WRN28_10000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][10000],

        untargeted_SVMOPT_WRN40_1000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][1000],
        untargeted_SVMOPT_WRN40_2000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][2000],
        untargeted_SVMOPT_WRN40_5000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][5000],
        untargeted_SVMOPT_WRN40_8000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][8000],
        untargeted_SVMOPT_WRN40_10000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][10000],

        targeted_SVMOPT_WRN40_1000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][1000],
        targeted_SVMOPT_WRN40_2000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][2000],
        targeted_SVMOPT_WRN40_5000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][5000],
        targeted_SVMOPT_WRN40_8000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][8000],
        targeted_SVMOPT_WRN40_10000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][10000],

        untargeted_GeoDA_pyramidnet272_1000=untargeted_result["pyramidnet272"]["GeoDA"][1000],
        untargeted_GeoDA_pyramidnet272_2000=untargeted_result["pyramidnet272"]["GeoDA"][2000],
        untargeted_GeoDA_pyramidnet272_5000=untargeted_result["pyramidnet272"]["GeoDA"][5000],
        untargeted_GeoDA_pyramidnet272_8000=untargeted_result["pyramidnet272"]["GeoDA"][8000],
        untargeted_GeoDA_pyramidnet272_10000=untargeted_result["pyramidnet272"]["GeoDA"][10000],

        targeted_GeoDA_pyramidnet272_1000=targeted_result["pyramidnet272"]["GeoDA"][1000],
        targeted_GeoDA_pyramidnet272_2000=targeted_result["pyramidnet272"]["GeoDA"][2000],
        targeted_GeoDA_pyramidnet272_5000=targeted_result["pyramidnet272"]["GeoDA"][5000],
        targeted_GeoDA_pyramidnet272_8000=targeted_result["pyramidnet272"]["GeoDA"][8000],
        targeted_GeoDA_pyramidnet272_10000=targeted_result["pyramidnet272"]["GeoDA"][10000],

        untargeted_GeoDA_gdas_1000=untargeted_result["gdas"]["GeoDA"][1000],
        untargeted_GeoDA_gdas_2000=untargeted_result["gdas"]["GeoDA"][2000],
        untargeted_GeoDA_gdas_5000=untargeted_result["gdas"]["GeoDA"][5000],
        untargeted_GeoDA_gdas_8000=untargeted_result["gdas"]["GeoDA"][8000],
        untargeted_GeoDA_gdas_10000=untargeted_result["gdas"]["GeoDA"][10000],

        targeted_GeoDA_gdas_1000=targeted_result["gdas"]["GeoDA"][1000],
        targeted_GeoDA_gdas_2000=targeted_result["gdas"]["GeoDA"][2000],
        targeted_GeoDA_gdas_5000=targeted_result["gdas"]["GeoDA"][5000],
        targeted_GeoDA_gdas_8000=targeted_result["gdas"]["GeoDA"][8000],
        targeted_GeoDA_gdas_10000=targeted_result["gdas"]["GeoDA"][10000],

        untargeted_GeoDA_WRN28_1000=untargeted_result["WRN-28-10-drop"]["GeoDA"][1000],
        untargeted_GeoDA_WRN28_2000=untargeted_result["WRN-28-10-drop"]["GeoDA"][2000],
        untargeted_GeoDA_WRN28_5000=untargeted_result["WRN-28-10-drop"]["GeoDA"][5000],
        untargeted_GeoDA_WRN28_8000=untargeted_result["WRN-28-10-drop"]["GeoDA"][8000],
        untargeted_GeoDA_WRN28_10000=untargeted_result["WRN-28-10-drop"]["GeoDA"][10000],

        targeted_GeoDA_WRN28_1000=targeted_result["WRN-28-10-drop"]["GeoDA"][1000],
        targeted_GeoDA_WRN28_2000=targeted_result["WRN-28-10-drop"]["GeoDA"][2000],
        targeted_GeoDA_WRN28_5000=targeted_result["WRN-28-10-drop"]["GeoDA"][5000],
        targeted_GeoDA_WRN28_8000=targeted_result["WRN-28-10-drop"]["GeoDA"][8000],
        targeted_GeoDA_WRN28_10000=targeted_result["WRN-28-10-drop"]["GeoDA"][10000],

        untargeted_GeoDA_WRN40_1000=untargeted_result["WRN-40-10-drop"]["GeoDA"][1000],
        untargeted_GeoDA_WRN40_2000=untargeted_result["WRN-40-10-drop"]["GeoDA"][2000],
        untargeted_GeoDA_WRN40_5000=untargeted_result["WRN-40-10-drop"]["GeoDA"][5000],
        untargeted_GeoDA_WRN40_8000=untargeted_result["WRN-40-10-drop"]["GeoDA"][8000],
        untargeted_GeoDA_WRN40_10000=untargeted_result["WRN-40-10-drop"]["GeoDA"][10000],

        targeted_GeoDA_WRN40_1000=targeted_result["WRN-40-10-drop"]["GeoDA"][1000],
        targeted_GeoDA_WRN40_2000=targeted_result["WRN-40-10-drop"]["GeoDA"][2000],
        targeted_GeoDA_WRN40_5000=targeted_result["WRN-40-10-drop"]["GeoDA"][5000],
        targeted_GeoDA_WRN40_8000=targeted_result["WRN-40-10-drop"]["GeoDA"][8000],
        targeted_GeoDA_WRN40_10000=targeted_result["WRN-40-10-drop"]["GeoDA"][10000],

        untargeted_RayS_pyramidnet272_1000=untargeted_result["pyramidnet272"]["RayS"][1000],
        untargeted_RayS_pyramidnet272_2000=untargeted_result["pyramidnet272"]["RayS"][2000],
        untargeted_RayS_pyramidnet272_5000=untargeted_result["pyramidnet272"]["RayS"][5000],
        untargeted_RayS_pyramidnet272_8000=untargeted_result["pyramidnet272"]["RayS"][8000],
        untargeted_RayS_pyramidnet272_10000=untargeted_result["pyramidnet272"]["RayS"][10000],

        targeted_RayS_pyramidnet272_1000=targeted_result["pyramidnet272"]["RayS"][1000],
        targeted_RayS_pyramidnet272_2000=targeted_result["pyramidnet272"]["RayS"][2000],
        targeted_RayS_pyramidnet272_5000=targeted_result["pyramidnet272"]["RayS"][5000],
        targeted_RayS_pyramidnet272_8000=targeted_result["pyramidnet272"]["RayS"][8000],
        targeted_RayS_pyramidnet272_10000=targeted_result["pyramidnet272"]["RayS"][10000],

        untargeted_RayS_gdas_1000=untargeted_result["gdas"]["RayS"][1000],
        untargeted_RayS_gdas_2000=untargeted_result["gdas"]["RayS"][2000],
        untargeted_RayS_gdas_5000=untargeted_result["gdas"]["RayS"][5000],
        untargeted_RayS_gdas_8000=untargeted_result["gdas"]["RayS"][8000],
        untargeted_RayS_gdas_10000=untargeted_result["gdas"]["RayS"][10000],

        targeted_RayS_gdas_1000=targeted_result["gdas"]["RayS"][1000],
        targeted_RayS_gdas_2000=targeted_result["gdas"]["RayS"][2000],
        targeted_RayS_gdas_5000=targeted_result["gdas"]["RayS"][5000],
        targeted_RayS_gdas_8000=targeted_result["gdas"]["RayS"][8000],
        targeted_RayS_gdas_10000=targeted_result["gdas"]["RayS"][10000],

        untargeted_RayS_WRN28_1000=untargeted_result["WRN-28-10-drop"]["RayS"][1000],
        untargeted_RayS_WRN28_2000=untargeted_result["WRN-28-10-drop"]["RayS"][2000],
        untargeted_RayS_WRN28_5000=untargeted_result["WRN-28-10-drop"]["RayS"][5000],
        untargeted_RayS_WRN28_8000=untargeted_result["WRN-28-10-drop"]["RayS"][8000],
        untargeted_RayS_WRN28_10000=untargeted_result["WRN-28-10-drop"]["RayS"][10000],

        targeted_RayS_WRN28_1000=targeted_result["WRN-28-10-drop"]["RayS"][1000],
        targeted_RayS_WRN28_2000=targeted_result["WRN-28-10-drop"]["RayS"][2000],
        targeted_RayS_WRN28_5000=targeted_result["WRN-28-10-drop"]["RayS"][5000],
        targeted_RayS_WRN28_8000=targeted_result["WRN-28-10-drop"]["RayS"][8000],
        targeted_RayS_WRN28_10000=targeted_result["WRN-28-10-drop"]["RayS"][10000],

        untargeted_RayS_WRN40_1000=untargeted_result["WRN-40-10-drop"]["RayS"][1000],
        untargeted_RayS_WRN40_2000=untargeted_result["WRN-40-10-drop"]["RayS"][2000],
        untargeted_RayS_WRN40_5000=untargeted_result["WRN-40-10-drop"]["RayS"][5000],
        untargeted_RayS_WRN40_8000=untargeted_result["WRN-40-10-drop"]["RayS"][8000],
        untargeted_RayS_WRN40_10000=untargeted_result["WRN-40-10-drop"]["RayS"][10000],

        targeted_RayS_WRN40_1000=targeted_result["WRN-40-10-drop"]["RayS"][1000],
        targeted_RayS_WRN40_2000=targeted_result["WRN-40-10-drop"]["RayS"][2000],
        targeted_RayS_WRN40_5000=targeted_result["WRN-40-10-drop"]["RayS"][5000],
        targeted_RayS_WRN40_8000=targeted_result["WRN-40-10-drop"]["RayS"][8000],
        targeted_RayS_WRN40_10000=targeted_result["WRN-40-10-drop"]["RayS"][10000],

        untargeted_HSJA_pyramidnet272_1000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_pyramidnet272_2000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_pyramidnet272_5000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_pyramidnet272_8000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_pyramidnet272_10000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_pyramidnet272_1000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_pyramidnet272_2000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_pyramidnet272_5000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_pyramidnet272_8000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_pyramidnet272_10000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][10000],

        untargeted_HSJA_gdas_1000=untargeted_result["gdas"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_gdas_2000=untargeted_result["gdas"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_gdas_5000=untargeted_result["gdas"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_gdas_8000=untargeted_result["gdas"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_gdas_10000=untargeted_result["gdas"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_gdas_1000=targeted_result["gdas"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_gdas_2000=targeted_result["gdas"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_gdas_5000=targeted_result["gdas"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_gdas_8000=targeted_result["gdas"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_gdas_10000=targeted_result["gdas"]["HopSkipJumpAttack"][10000],

        untargeted_HSJA_WRN28_1000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_WRN28_2000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_WRN28_5000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_WRN28_8000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_WRN28_10000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_WRN28_1000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_WRN28_2000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_WRN28_5000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_WRN28_8000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_WRN28_10000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][10000],

        untargeted_HSJA_WRN40_1000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_WRN40_2000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_WRN40_5000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_WRN40_8000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_WRN40_10000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_WRN40_1000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_WRN40_2000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_WRN40_5000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_WRN40_8000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_WRN40_10000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][10000],

        untargeted_Tangent_pyramidnet272_1000=untargeted_result["pyramidnet272"]["Tangent Attack"][1000],
        untargeted_Tangent_pyramidnet272_2000=untargeted_result["pyramidnet272"]["Tangent Attack"][2000],
        untargeted_Tangent_pyramidnet272_5000=untargeted_result["pyramidnet272"]["Tangent Attack"][5000],
        untargeted_Tangent_pyramidnet272_8000=untargeted_result["pyramidnet272"]["Tangent Attack"][8000],
        untargeted_Tangent_pyramidnet272_10000=untargeted_result["pyramidnet272"]["Tangent Attack"][10000],

        targeted_Tangent_pyramidnet272_1000=targeted_result["pyramidnet272"]["Tangent Attack"][1000],
        targeted_Tangent_pyramidnet272_2000=targeted_result["pyramidnet272"]["Tangent Attack"][2000],
        targeted_Tangent_pyramidnet272_5000=targeted_result["pyramidnet272"]["Tangent Attack"][5000],
        targeted_Tangent_pyramidnet272_8000=targeted_result["pyramidnet272"]["Tangent Attack"][8000],
        targeted_Tangent_pyramidnet272_10000=targeted_result["pyramidnet272"]["Tangent Attack"][10000],

        untargeted_Tangent_gdas_1000=untargeted_result["gdas"]["Tangent Attack"][1000],
        untargeted_Tangent_gdas_2000=untargeted_result["gdas"]["Tangent Attack"][2000],
        untargeted_Tangent_gdas_5000=untargeted_result["gdas"]["Tangent Attack"][5000],
        untargeted_Tangent_gdas_8000=untargeted_result["gdas"]["Tangent Attack"][8000],
        untargeted_Tangent_gdas_10000=untargeted_result["gdas"]["Tangent Attack"][10000],

        targeted_Tangent_gdas_1000=targeted_result["gdas"]["Tangent Attack"][1000],
        targeted_Tangent_gdas_2000=targeted_result["gdas"]["Tangent Attack"][2000],
        targeted_Tangent_gdas_5000=targeted_result["gdas"]["Tangent Attack"][5000],
        targeted_Tangent_gdas_8000=targeted_result["gdas"]["Tangent Attack"][8000],
        targeted_Tangent_gdas_10000=targeted_result["gdas"]["Tangent Attack"][10000],

        untargeted_Tangent_WRN28_1000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][1000],
        untargeted_Tangent_WRN28_2000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][2000],
        untargeted_Tangent_WRN28_5000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][5000],
        untargeted_Tangent_WRN28_8000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][8000],
        untargeted_Tangent_WRN28_10000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][10000],

        targeted_Tangent_WRN28_1000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][1000],
        targeted_Tangent_WRN28_2000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][2000],
        targeted_Tangent_WRN28_5000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][5000],
        targeted_Tangent_WRN28_8000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][8000],
        targeted_Tangent_WRN28_10000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][10000],

        untargeted_Tangent_WRN40_1000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][1000],
        untargeted_Tangent_WRN40_2000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][2000],
        untargeted_Tangent_WRN40_5000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][5000],
        untargeted_Tangent_WRN40_8000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][8000],
        untargeted_Tangent_WRN40_10000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][10000],

        targeted_Tangent_WRN40_1000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][1000],
        targeted_Tangent_WRN40_2000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][2000],
        targeted_Tangent_WRN40_5000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][5000],
        targeted_Tangent_WRN40_8000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][8000],
        targeted_Tangent_WRN40_10000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][10000],
    )
    )

def draw_wide_table_CIFAR_with_boundary(untargeted_result, targeted_result):

    print("""
                    & Boundary & {targeted_Boundary_pyramidnet272_1000} & {targeted_Boundary_pyramidnet272_2000} & {targeted_Boundary_pyramidnet272_5000} & {targeted_Boundary_pyramidnet272_8000} & {targeted_Boundary_pyramidnet272_10000} & {untargeted_Boundary_pyramidnet272_1000} & {untargeted_Boundary_pyramidnet272_2000} & {untargeted_Boundary_pyramidnet272_5000} & {untargeted_Boundary_pyramidnet272_8000} & {untargeted_Boundary_pyramidnet272_10000}  \\\\
                    & Sign-OPT & {targeted_SignOPT_pyramidnet272_1000} & {targeted_SignOPT_pyramidnet272_2000} & {targeted_SignOPT_pyramidnet272_5000} & {targeted_SignOPT_pyramidnet272_8000} & {targeted_SignOPT_pyramidnet272_10000} & {untargeted_SignOPT_pyramidnet272_1000} & {untargeted_SignOPT_pyramidnet272_2000} & {untargeted_SignOPT_pyramidnet272_5000} & {untargeted_SignOPT_pyramidnet272_8000} & {untargeted_SignOPT_pyramidnet272_10000}  \\\\
                    & SVM-OPT & {targeted_SVMOPT_pyramidnet272_1000} & {targeted_SVMOPT_pyramidnet272_2000} & {targeted_SVMOPT_pyramidnet272_5000} & {targeted_SVMOPT_pyramidnet272_8000} & {targeted_SVMOPT_pyramidnet272_10000} & {untargeted_SVMOPT_pyramidnet272_1000} & {untargeted_SVMOPT_pyramidnet272_2000} & {untargeted_SVMOPT_pyramidnet272_5000} & {untargeted_SVMOPT_pyramidnet272_8000} & {untargeted_SVMOPT_pyramidnet272_10000}  \\\\
                    & HSJA & {targeted_HSJA_pyramidnet272_1000} & {targeted_HSJA_pyramidnet272_2000} & {targeted_HSJA_pyramidnet272_5000} & {targeted_HSJA_pyramidnet272_8000} & {targeted_HSJA_pyramidnet272_10000} & {untargeted_HSJA_pyramidnet272_1000} & {untargeted_HSJA_pyramidnet272_2000} & {untargeted_HSJA_pyramidnet272_5000} & {untargeted_HSJA_pyramidnet272_8000} & {untargeted_HSJA_pyramidnet272_10000}  \\\\
                    & Ours & {targeted_Tangent_pyramidnet272_1000} & {targeted_Tangent_pyramidnet272_2000} & {targeted_Tangent_pyramidnet272_5000} & {targeted_Tangent_pyramidnet272_8000} & {targeted_Tangent_pyramidnet272_10000} & {untargeted_Tangent_pyramidnet272_1000} & {untargeted_Tangent_pyramidnet272_2000} & {untargeted_Tangent_pyramidnet272_5000} & {untargeted_Tangent_pyramidnet272_8000} & {untargeted_Tangent_pyramidnet272_10000}  \\\\
                    \\midrule
                    & Boundary & {targeted_Boundary_gdas_1000} & {targeted_Boundary_gdas_2000} & {targeted_Boundary_gdas_5000} & {targeted_Boundary_gdas_8000} & {targeted_Boundary_gdas_10000} & {untargeted_Boundary_gdas_1000} & {untargeted_Boundary_gdas_2000} & {untargeted_Boundary_gdas_5000} & {untargeted_Boundary_gdas_8000} & {untargeted_Boundary_gdas_10000}  \\\\
                    & Sign-OPT & {targeted_SignOPT_gdas_1000} & {targeted_SignOPT_gdas_2000} & {targeted_SignOPT_gdas_5000} & {targeted_SignOPT_gdas_8000} & {targeted_SignOPT_gdas_10000} & {untargeted_SignOPT_gdas_1000} & {untargeted_SignOPT_gdas_2000} & {untargeted_SignOPT_gdas_5000} & {untargeted_SignOPT_gdas_8000} & {untargeted_SignOPT_gdas_10000}  \\\\
                    & SVM-OPT & {targeted_SVMOPT_gdas_1000} & {targeted_SVMOPT_gdas_2000} & {targeted_SVMOPT_gdas_5000} & {targeted_SVMOPT_gdas_8000} & {targeted_SVMOPT_gdas_10000} & {untargeted_SVMOPT_gdas_1000} & {untargeted_SVMOPT_gdas_2000} & {untargeted_SVMOPT_gdas_5000} & {untargeted_SVMOPT_gdas_8000} & {untargeted_SVMOPT_gdas_10000}  \\\\
                    & HSJA & {targeted_HSJA_gdas_1000} & {targeted_HSJA_gdas_2000} & {targeted_HSJA_gdas_5000} & {targeted_HSJA_gdas_8000} & {targeted_HSJA_gdas_10000} & {untargeted_HSJA_gdas_1000} & {untargeted_HSJA_gdas_2000} & {untargeted_HSJA_gdas_5000} & {untargeted_HSJA_gdas_8000} & {untargeted_HSJA_gdas_10000}  \\\\
                    & Ours & {targeted_Tangent_gdas_1000} & {targeted_Tangent_gdas_2000} & {targeted_Tangent_gdas_5000} & {targeted_Tangent_gdas_8000} & {targeted_Tangent_gdas_10000} & {untargeted_Tangent_gdas_1000} & {untargeted_Tangent_gdas_2000} & {untargeted_Tangent_gdas_5000} & {untargeted_Tangent_gdas_8000} & {untargeted_Tangent_gdas_10000}  \\\\
                    \\midrule
                    & Boundary & {targeted_Boundary_WRN28_1000} & {targeted_Boundary_WRN28_2000} & {targeted_Boundary_WRN28_5000} & {targeted_Boundary_WRN28_8000} & {targeted_Boundary_WRN28_10000} & {untargeted_Boundary_WRN28_1000} & {untargeted_Boundary_WRN28_2000} & {untargeted_Boundary_WRN28_5000} & {untargeted_Boundary_WRN28_8000} & {untargeted_Boundary_WRN28_10000}  \\\\
                    & Sign-OPT & {targeted_SignOPT_WRN28_1000} & {targeted_SignOPT_WRN28_2000} & {targeted_SignOPT_WRN28_5000} & {targeted_SignOPT_WRN28_8000} & {targeted_SignOPT_WRN28_10000} & {untargeted_SignOPT_WRN28_1000} & {untargeted_SignOPT_WRN28_2000} & {untargeted_SignOPT_WRN28_5000} & {untargeted_SignOPT_WRN28_8000} & {untargeted_SignOPT_WRN28_10000}  \\\\
                    & SVM-OPT & {targeted_SVMOPT_WRN28_1000} & {targeted_SVMOPT_WRN28_2000} & {targeted_SVMOPT_WRN28_5000} & {targeted_SVMOPT_WRN28_8000} & {targeted_SVMOPT_WRN28_10000} & {untargeted_SVMOPT_WRN28_1000} & {untargeted_SVMOPT_WRN28_2000} & {untargeted_SVMOPT_WRN28_5000} & {untargeted_SVMOPT_WRN28_8000} & {untargeted_SVMOPT_WRN28_10000}  \\\\
                    & HSJA & {targeted_HSJA_WRN28_1000} & {targeted_HSJA_WRN28_2000} & {targeted_HSJA_WRN28_5000} & {targeted_HSJA_WRN28_8000} & {targeted_HSJA_WRN28_10000} & {untargeted_HSJA_WRN28_1000} & {untargeted_HSJA_WRN28_2000} & {untargeted_HSJA_WRN28_5000} & {untargeted_HSJA_WRN28_8000} & {untargeted_HSJA_WRN28_10000}  \\\\
                    & Ours & {targeted_Tangent_WRN28_1000} & {targeted_Tangent_WRN28_2000} & {targeted_Tangent_WRN28_5000} & {targeted_Tangent_WRN28_8000} & {targeted_Tangent_WRN28_10000} & {untargeted_Tangent_WRN28_1000} & {untargeted_Tangent_WRN28_2000} & {untargeted_Tangent_WRN28_5000} & {untargeted_Tangent_WRN28_8000} & {untargeted_Tangent_WRN28_10000}  \\\\
                    \\midrule
                    & Boundary & {targeted_Boundary_WRN40_1000} & {targeted_Boundary_WRN40_2000} & {targeted_Boundary_WRN40_5000} & {targeted_Boundary_WRN40_8000} & {targeted_Boundary_WRN40_10000} & {untargeted_Boundary_WRN40_1000} & {untargeted_Boundary_WRN40_2000} & {untargeted_Boundary_WRN40_5000} & {untargeted_Boundary_WRN40_8000} & {untargeted_Boundary_WRN40_10000}  \\\\
                    & Sign-OPT & {targeted_SignOPT_WRN40_1000} & {targeted_SignOPT_WRN40_2000} & {targeted_SignOPT_WRN40_5000} & {targeted_SignOPT_WRN40_8000} & {targeted_SignOPT_WRN40_10000} & {untargeted_SignOPT_WRN40_1000} & {untargeted_SignOPT_WRN40_2000} & {untargeted_SignOPT_WRN40_5000} & {untargeted_SignOPT_WRN40_8000} & {untargeted_SignOPT_WRN40_10000}  \\\\
                    & SVM-OPT & {targeted_SVMOPT_WRN40_1000} & {targeted_SVMOPT_WRN40_2000} & {targeted_SVMOPT_WRN40_5000} & {targeted_SVMOPT_WRN40_8000} & {targeted_SVMOPT_WRN40_10000} & {untargeted_SVMOPT_WRN40_1000} & {untargeted_SVMOPT_WRN40_2000} & {untargeted_SVMOPT_WRN40_5000} & {untargeted_SVMOPT_WRN40_8000} & {untargeted_SVMOPT_WRN40_10000}  \\\\
                    & HSJA & {targeted_HSJA_WRN40_1000} & {targeted_HSJA_WRN40_2000} & {targeted_HSJA_WRN40_5000} & {targeted_HSJA_WRN40_8000} & {targeted_HSJA_WRN40_10000} & {untargeted_HSJA_WRN40_1000} & {untargeted_HSJA_WRN40_2000} & {untargeted_HSJA_WRN40_5000} & {untargeted_HSJA_WRN40_8000} & {untargeted_HSJA_WRN40_10000}  \\\\
                    & Ours & {targeted_Tangent_WRN40_1000} & {targeted_Tangent_WRN40_2000} & {targeted_Tangent_WRN40_5000} & {targeted_Tangent_WRN40_8000} & {targeted_Tangent_WRN40_10000} & {untargeted_Tangent_WRN40_1000} & {untargeted_Tangent_WRN40_2000} & {untargeted_Tangent_WRN40_5000} & {untargeted_Tangent_WRN40_8000} & {untargeted_Tangent_WRN40_10000}  \\\\
                        """.format(

        untargeted_SignOPT_pyramidnet272_1000=untargeted_result["pyramidnet272"]["Sign-OPT"][1000],
        untargeted_SignOPT_pyramidnet272_2000=untargeted_result["pyramidnet272"]["Sign-OPT"][2000],
        untargeted_SignOPT_pyramidnet272_5000=untargeted_result["pyramidnet272"]["Sign-OPT"][5000],
        untargeted_SignOPT_pyramidnet272_8000=untargeted_result["pyramidnet272"]["Sign-OPT"][8000],
        untargeted_SignOPT_pyramidnet272_10000=untargeted_result["pyramidnet272"]["Sign-OPT"][10000],

        targeted_SignOPT_pyramidnet272_1000=targeted_result["pyramidnet272"]["Sign-OPT"][1000],
        targeted_SignOPT_pyramidnet272_2000=targeted_result["pyramidnet272"]["Sign-OPT"][2000],
        targeted_SignOPT_pyramidnet272_5000=targeted_result["pyramidnet272"]["Sign-OPT"][5000],
        targeted_SignOPT_pyramidnet272_8000=targeted_result["pyramidnet272"]["Sign-OPT"][8000],
        targeted_SignOPT_pyramidnet272_10000=targeted_result["pyramidnet272"]["Sign-OPT"][10000],

        untargeted_SignOPT_gdas_1000=untargeted_result["gdas"]["Sign-OPT"][1000],
        untargeted_SignOPT_gdas_2000=untargeted_result["gdas"]["Sign-OPT"][2000],
        untargeted_SignOPT_gdas_5000=untargeted_result["gdas"]["Sign-OPT"][5000],
        untargeted_SignOPT_gdas_8000=untargeted_result["gdas"]["Sign-OPT"][8000],
        untargeted_SignOPT_gdas_10000=untargeted_result["gdas"]["Sign-OPT"][10000],

        targeted_SignOPT_gdas_1000=targeted_result["gdas"]["Sign-OPT"][1000],
        targeted_SignOPT_gdas_2000=targeted_result["gdas"]["Sign-OPT"][2000],
        targeted_SignOPT_gdas_5000=targeted_result["gdas"]["Sign-OPT"][5000],
        targeted_SignOPT_gdas_8000=targeted_result["gdas"]["Sign-OPT"][8000],
        targeted_SignOPT_gdas_10000=targeted_result["gdas"]["Sign-OPT"][10000],

        untargeted_SignOPT_WRN28_1000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][1000],
        untargeted_SignOPT_WRN28_2000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][2000],
        untargeted_SignOPT_WRN28_5000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][5000],
        untargeted_SignOPT_WRN28_8000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][8000],
        untargeted_SignOPT_WRN28_10000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][10000],

        targeted_SignOPT_WRN28_1000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][1000],
        targeted_SignOPT_WRN28_2000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][2000],
        targeted_SignOPT_WRN28_5000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][5000],
        targeted_SignOPT_WRN28_8000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][8000],
        targeted_SignOPT_WRN28_10000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][10000],

        untargeted_SignOPT_WRN40_1000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][1000],
        untargeted_SignOPT_WRN40_2000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][2000],
        untargeted_SignOPT_WRN40_5000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][5000],
        untargeted_SignOPT_WRN40_8000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][8000],
        untargeted_SignOPT_WRN40_10000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][10000],

        targeted_SignOPT_WRN40_1000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][1000],
        targeted_SignOPT_WRN40_2000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][2000],
        targeted_SignOPT_WRN40_5000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][5000],
        targeted_SignOPT_WRN40_8000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][8000],
        targeted_SignOPT_WRN40_10000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][10000],

        untargeted_SVMOPT_pyramidnet272_1000=untargeted_result["pyramidnet272"]["SVM-OPT"][1000],
        untargeted_SVMOPT_pyramidnet272_2000=untargeted_result["pyramidnet272"]["SVM-OPT"][2000],
        untargeted_SVMOPT_pyramidnet272_5000=untargeted_result["pyramidnet272"]["SVM-OPT"][5000],
        untargeted_SVMOPT_pyramidnet272_8000=untargeted_result["pyramidnet272"]["SVM-OPT"][8000],
        untargeted_SVMOPT_pyramidnet272_10000=untargeted_result["pyramidnet272"]["SVM-OPT"][10000],

        targeted_SVMOPT_pyramidnet272_1000=targeted_result["pyramidnet272"]["SVM-OPT"][1000],
        targeted_SVMOPT_pyramidnet272_2000=targeted_result["pyramidnet272"]["SVM-OPT"][2000],
        targeted_SVMOPT_pyramidnet272_5000=targeted_result["pyramidnet272"]["SVM-OPT"][5000],
        targeted_SVMOPT_pyramidnet272_8000=targeted_result["pyramidnet272"]["SVM-OPT"][8000],
        targeted_SVMOPT_pyramidnet272_10000=targeted_result["pyramidnet272"]["SVM-OPT"][10000],

        untargeted_SVMOPT_gdas_1000=untargeted_result["gdas"]["SVM-OPT"][1000],
        untargeted_SVMOPT_gdas_2000=untargeted_result["gdas"]["SVM-OPT"][2000],
        untargeted_SVMOPT_gdas_5000=untargeted_result["gdas"]["SVM-OPT"][5000],
        untargeted_SVMOPT_gdas_8000=untargeted_result["gdas"]["SVM-OPT"][8000],
        untargeted_SVMOPT_gdas_10000=untargeted_result["gdas"]["SVM-OPT"][10000],

        targeted_SVMOPT_gdas_1000=targeted_result["gdas"]["SVM-OPT"][1000],
        targeted_SVMOPT_gdas_2000=targeted_result["gdas"]["SVM-OPT"][2000],
        targeted_SVMOPT_gdas_5000=targeted_result["gdas"]["SVM-OPT"][5000],
        targeted_SVMOPT_gdas_8000=targeted_result["gdas"]["SVM-OPT"][8000],
        targeted_SVMOPT_gdas_10000=targeted_result["gdas"]["SVM-OPT"][10000],

        untargeted_SVMOPT_WRN28_1000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][1000],
        untargeted_SVMOPT_WRN28_2000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][2000],
        untargeted_SVMOPT_WRN28_5000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][5000],
        untargeted_SVMOPT_WRN28_8000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][8000],
        untargeted_SVMOPT_WRN28_10000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][10000],

        targeted_SVMOPT_WRN28_1000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][1000],
        targeted_SVMOPT_WRN28_2000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][2000],
        targeted_SVMOPT_WRN28_5000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][5000],
        targeted_SVMOPT_WRN28_8000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][8000],
        targeted_SVMOPT_WRN28_10000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][10000],

        untargeted_SVMOPT_WRN40_1000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][1000],
        untargeted_SVMOPT_WRN40_2000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][2000],
        untargeted_SVMOPT_WRN40_5000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][5000],
        untargeted_SVMOPT_WRN40_8000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][8000],
        untargeted_SVMOPT_WRN40_10000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][10000],

        targeted_SVMOPT_WRN40_1000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][1000],
        targeted_SVMOPT_WRN40_2000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][2000],
        targeted_SVMOPT_WRN40_5000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][5000],
        targeted_SVMOPT_WRN40_8000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][8000],
        targeted_SVMOPT_WRN40_10000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][10000],

        untargeted_Boundary_pyramidnet272_1000=untargeted_result["pyramidnet272"]["Boundary Attack"][1000],
        untargeted_Boundary_pyramidnet272_2000=untargeted_result["pyramidnet272"]["Boundary Attack"][2000],
        untargeted_Boundary_pyramidnet272_5000=untargeted_result["pyramidnet272"]["Boundary Attack"][5000],
        untargeted_Boundary_pyramidnet272_8000=untargeted_result["pyramidnet272"]["Boundary Attack"][8000],
        untargeted_Boundary_pyramidnet272_10000=untargeted_result["pyramidnet272"]["Boundary Attack"][10000],

        targeted_Boundary_pyramidnet272_1000=targeted_result["pyramidnet272"]["Boundary Attack"][1000],
        targeted_Boundary_pyramidnet272_2000=targeted_result["pyramidnet272"]["Boundary Attack"][2000],
        targeted_Boundary_pyramidnet272_5000=targeted_result["pyramidnet272"]["Boundary Attack"][5000],
        targeted_Boundary_pyramidnet272_8000=targeted_result["pyramidnet272"]["Boundary Attack"][8000],
        targeted_Boundary_pyramidnet272_10000=targeted_result["pyramidnet272"]["Boundary Attack"][10000],

        untargeted_Boundary_gdas_1000=untargeted_result["gdas"]["Boundary Attack"][1000],
        untargeted_Boundary_gdas_2000=untargeted_result["gdas"]["Boundary Attack"][2000],
        untargeted_Boundary_gdas_5000=untargeted_result["gdas"]["Boundary Attack"][5000],
        untargeted_Boundary_gdas_8000=untargeted_result["gdas"]["Boundary Attack"][8000],
        untargeted_Boundary_gdas_10000=untargeted_result["gdas"]["Boundary Attack"][10000],

        targeted_Boundary_gdas_1000=targeted_result["gdas"]["Boundary Attack"][1000],
        targeted_Boundary_gdas_2000=targeted_result["gdas"]["Boundary Attack"][2000],
        targeted_Boundary_gdas_5000=targeted_result["gdas"]["Boundary Attack"][5000],
        targeted_Boundary_gdas_8000=targeted_result["gdas"]["Boundary Attack"][8000],
        targeted_Boundary_gdas_10000=targeted_result["gdas"]["Boundary Attack"][10000],

        untargeted_Boundary_WRN28_1000=untargeted_result["WRN-28-10-drop"]["Boundary Attack"][1000],
        untargeted_Boundary_WRN28_2000=untargeted_result["WRN-28-10-drop"]["Boundary Attack"][2000],
        untargeted_Boundary_WRN28_5000=untargeted_result["WRN-28-10-drop"]["Boundary Attack"][5000],
        untargeted_Boundary_WRN28_8000=untargeted_result["WRN-28-10-drop"]["Boundary Attack"][8000],
        untargeted_Boundary_WRN28_10000=untargeted_result["WRN-28-10-drop"]["Boundary Attack"][10000],

        targeted_Boundary_WRN28_1000=targeted_result["WRN-28-10-drop"]["Boundary Attack"][1000],
        targeted_Boundary_WRN28_2000=targeted_result["WRN-28-10-drop"]["Boundary Attack"][2000],
        targeted_Boundary_WRN28_5000=targeted_result["WRN-28-10-drop"]["Boundary Attack"][5000],
        targeted_Boundary_WRN28_8000=targeted_result["WRN-28-10-drop"]["Boundary Attack"][8000],
        targeted_Boundary_WRN28_10000=targeted_result["WRN-28-10-drop"]["Boundary Attack"][10000],

        untargeted_Boundary_WRN40_1000=untargeted_result["WRN-40-10-drop"]["Boundary Attack"][1000],
        untargeted_Boundary_WRN40_2000=untargeted_result["WRN-40-10-drop"]["Boundary Attack"][2000],
        untargeted_Boundary_WRN40_5000=untargeted_result["WRN-40-10-drop"]["Boundary Attack"][5000],
        untargeted_Boundary_WRN40_8000=untargeted_result["WRN-40-10-drop"]["Boundary Attack"][8000],
        untargeted_Boundary_WRN40_10000=untargeted_result["WRN-40-10-drop"]["Boundary Attack"][10000],

        targeted_Boundary_WRN40_1000=targeted_result["WRN-40-10-drop"]["Boundary Attack"][1000],
        targeted_Boundary_WRN40_2000=targeted_result["WRN-40-10-drop"]["Boundary Attack"][2000],
        targeted_Boundary_WRN40_5000=targeted_result["WRN-40-10-drop"]["Boundary Attack"][5000],
        targeted_Boundary_WRN40_8000=targeted_result["WRN-40-10-drop"]["Boundary Attack"][8000],
        targeted_Boundary_WRN40_10000=targeted_result["WRN-40-10-drop"]["Boundary Attack"][10000],

        untargeted_HSJA_pyramidnet272_1000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_pyramidnet272_2000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_pyramidnet272_5000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_pyramidnet272_8000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_pyramidnet272_10000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_pyramidnet272_1000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_pyramidnet272_2000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_pyramidnet272_5000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_pyramidnet272_8000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_pyramidnet272_10000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][10000],

        untargeted_HSJA_gdas_1000=untargeted_result["gdas"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_gdas_2000=untargeted_result["gdas"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_gdas_5000=untargeted_result["gdas"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_gdas_8000=untargeted_result["gdas"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_gdas_10000=untargeted_result["gdas"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_gdas_1000=targeted_result["gdas"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_gdas_2000=targeted_result["gdas"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_gdas_5000=targeted_result["gdas"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_gdas_8000=targeted_result["gdas"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_gdas_10000=targeted_result["gdas"]["HopSkipJumpAttack"][10000],

        untargeted_HSJA_WRN28_1000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_WRN28_2000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_WRN28_5000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_WRN28_8000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_WRN28_10000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_WRN28_1000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_WRN28_2000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_WRN28_5000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_WRN28_8000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_WRN28_10000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][10000],

        untargeted_HSJA_WRN40_1000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_WRN40_2000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_WRN40_5000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_WRN40_8000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_WRN40_10000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_WRN40_1000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_WRN40_2000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_WRN40_5000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_WRN40_8000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_WRN40_10000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][10000],

        untargeted_Tangent_pyramidnet272_1000=untargeted_result["pyramidnet272"]["Tangent Attack"][1000],
        untargeted_Tangent_pyramidnet272_2000=untargeted_result["pyramidnet272"]["Tangent Attack"][2000],
        untargeted_Tangent_pyramidnet272_5000=untargeted_result["pyramidnet272"]["Tangent Attack"][5000],
        untargeted_Tangent_pyramidnet272_8000=untargeted_result["pyramidnet272"]["Tangent Attack"][8000],
        untargeted_Tangent_pyramidnet272_10000=untargeted_result["pyramidnet272"]["Tangent Attack"][10000],

        targeted_Tangent_pyramidnet272_1000=targeted_result["pyramidnet272"]["Tangent Attack"][1000],
        targeted_Tangent_pyramidnet272_2000=targeted_result["pyramidnet272"]["Tangent Attack"][2000],
        targeted_Tangent_pyramidnet272_5000=targeted_result["pyramidnet272"]["Tangent Attack"][5000],
        targeted_Tangent_pyramidnet272_8000=targeted_result["pyramidnet272"]["Tangent Attack"][8000],
        targeted_Tangent_pyramidnet272_10000=targeted_result["pyramidnet272"]["Tangent Attack"][10000],

        untargeted_Tangent_gdas_1000=untargeted_result["gdas"]["Tangent Attack"][1000],
        untargeted_Tangent_gdas_2000=untargeted_result["gdas"]["Tangent Attack"][2000],
        untargeted_Tangent_gdas_5000=untargeted_result["gdas"]["Tangent Attack"][5000],
        untargeted_Tangent_gdas_8000=untargeted_result["gdas"]["Tangent Attack"][8000],
        untargeted_Tangent_gdas_10000=untargeted_result["gdas"]["Tangent Attack"][10000],

        targeted_Tangent_gdas_1000=targeted_result["gdas"]["Tangent Attack"][1000],
        targeted_Tangent_gdas_2000=targeted_result["gdas"]["Tangent Attack"][2000],
        targeted_Tangent_gdas_5000=targeted_result["gdas"]["Tangent Attack"][5000],
        targeted_Tangent_gdas_8000=targeted_result["gdas"]["Tangent Attack"][8000],
        targeted_Tangent_gdas_10000=targeted_result["gdas"]["Tangent Attack"][10000],

        untargeted_Tangent_WRN28_1000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][1000],
        untargeted_Tangent_WRN28_2000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][2000],
        untargeted_Tangent_WRN28_5000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][5000],
        untargeted_Tangent_WRN28_8000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][8000],
        untargeted_Tangent_WRN28_10000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][10000],

        targeted_Tangent_WRN28_1000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][1000],
        targeted_Tangent_WRN28_2000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][2000],
        targeted_Tangent_WRN28_5000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][5000],
        targeted_Tangent_WRN28_8000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][8000],
        targeted_Tangent_WRN28_10000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][10000],

        untargeted_Tangent_WRN40_1000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][1000],
        untargeted_Tangent_WRN40_2000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][2000],
        untargeted_Tangent_WRN40_5000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][5000],
        untargeted_Tangent_WRN40_8000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][8000],
        untargeted_Tangent_WRN40_10000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][10000],

        targeted_Tangent_WRN40_1000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][1000],
        targeted_Tangent_WRN40_2000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][2000],
        targeted_Tangent_WRN40_5000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][5000],
        targeted_Tangent_WRN40_8000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][8000],
        targeted_Tangent_WRN40_10000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][10000],
    )
    )


def draw_wide_table_CIFAR_add_init_batch_size_30(untargeted_result, targeted_result):
    print("""
                        & Boundary & {targeted_Boundary_pyramidnet272_1000} & {targeted_Boundary_pyramidnet272_2000} & {targeted_Boundary_pyramidnet272_5000} & {targeted_Boundary_pyramidnet272_8000} & {targeted_Boundary_pyramidnet272_10000} & {untargeted_Boundary_pyramidnet272_1000} & {untargeted_Boundary_pyramidnet272_2000} & {untargeted_Boundary_pyramidnet272_5000} & {untargeted_Boundary_pyramidnet272_8000} & {untargeted_Boundary_pyramidnet272_10000}  \\\\
                        & Sign-OPT & {targeted_SignOPT_pyramidnet272_1000} & {targeted_SignOPT_pyramidnet272_2000} & {targeted_SignOPT_pyramidnet272_5000} & {targeted_SignOPT_pyramidnet272_8000} & {targeted_SignOPT_pyramidnet272_10000} & {untargeted_SignOPT_pyramidnet272_1000} & {untargeted_SignOPT_pyramidnet272_2000} & {untargeted_SignOPT_pyramidnet272_5000} & {untargeted_SignOPT_pyramidnet272_8000} & {untargeted_SignOPT_pyramidnet272_10000}  \\\\
                        & SVM-OPT & {targeted_SVMOPT_pyramidnet272_1000} & {targeted_SVMOPT_pyramidnet272_2000} & {targeted_SVMOPT_pyramidnet272_5000} & {targeted_SVMOPT_pyramidnet272_8000} & {targeted_SVMOPT_pyramidnet272_10000} & {untargeted_SVMOPT_pyramidnet272_1000} & {untargeted_SVMOPT_pyramidnet272_2000} & {untargeted_SVMOPT_pyramidnet272_5000} & {untargeted_SVMOPT_pyramidnet272_8000} & {untargeted_SVMOPT_pyramidnet272_10000}  \\\\
                        & HSJA ($B_0=100$) & {targeted_HSJA_pyramidnet272_1000} & {targeted_HSJA_pyramidnet272_2000} & {targeted_HSJA_pyramidnet272_5000} & {targeted_HSJA_pyramidnet272_8000} & {targeted_HSJA_pyramidnet272_10000} & {untargeted_HSJA_pyramidnet272_1000} & {untargeted_HSJA_pyramidnet272_2000} & {untargeted_HSJA_pyramidnet272_5000} & {untargeted_HSJA_pyramidnet272_8000} & {untargeted_HSJA_pyramidnet272_10000}  \\\\
                        & Ours ($B_0=100$) & {targeted_Tangent_pyramidnet272_1000} & {targeted_Tangent_pyramidnet272_2000} & {targeted_Tangent_pyramidnet272_5000} & {targeted_Tangent_pyramidnet272_8000} & {targeted_Tangent_pyramidnet272_10000} & {untargeted_Tangent_pyramidnet272_1000} & {untargeted_Tangent_pyramidnet272_2000} & {untargeted_Tangent_pyramidnet272_5000} & {untargeted_Tangent_pyramidnet272_8000} & {untargeted_Tangent_pyramidnet272_10000}  \\\\
                        & HSJA ($B_0=30$) & {targeted_HSJA30_pyramidnet272_1000} & {targeted_HSJA30_pyramidnet272_2000} & {targeted_HSJA30_pyramidnet272_5000} & {targeted_HSJA30_pyramidnet272_8000} & {targeted_HSJA30_pyramidnet272_10000} & {untargeted_HSJA30_pyramidnet272_1000} & {untargeted_HSJA30_pyramidnet272_2000} & {untargeted_HSJA30_pyramidnet272_5000} & {untargeted_HSJA30_pyramidnet272_8000} & {untargeted_HSJA30_pyramidnet272_10000}  \\\\
                        & Ours ($B_0=30$) & {targeted_Tangent30_pyramidnet272_1000} & {targeted_Tangent30_pyramidnet272_2000} & {targeted_Tangent30_pyramidnet272_5000} & {targeted_Tangent30_pyramidnet272_8000} & {targeted_Tangent30_pyramidnet272_10000} & {untargeted_Tangent30_pyramidnet272_1000} & {untargeted_Tangent30_pyramidnet272_2000} & {untargeted_Tangent30_pyramidnet272_5000} & {untargeted_Tangent30_pyramidnet272_8000} & {untargeted_Tangent30_pyramidnet272_10000}  \\\\
                        \\midrule
                        & Boundary & {targeted_Boundary_gdas_1000} & {targeted_Boundary_gdas_2000} & {targeted_Boundary_gdas_5000} & {targeted_Boundary_gdas_8000} & {targeted_Boundary_gdas_10000} & {untargeted_Boundary_gdas_1000} & {untargeted_Boundary_gdas_2000} & {untargeted_Boundary_gdas_5000} & {untargeted_Boundary_gdas_8000} & {untargeted_Boundary_gdas_10000}  \\\\
                        & Sign-OPT & {targeted_SignOPT_gdas_1000} & {targeted_SignOPT_gdas_2000} & {targeted_SignOPT_gdas_5000} & {targeted_SignOPT_gdas_8000} & {targeted_SignOPT_gdas_10000} & {untargeted_SignOPT_gdas_1000} & {untargeted_SignOPT_gdas_2000} & {untargeted_SignOPT_gdas_5000} & {untargeted_SignOPT_gdas_8000} & {untargeted_SignOPT_gdas_10000}  \\\\
                        & SVM-OPT & {targeted_SVMOPT_gdas_1000} & {targeted_SVMOPT_gdas_2000} & {targeted_SVMOPT_gdas_5000} & {targeted_SVMOPT_gdas_8000} & {targeted_SVMOPT_gdas_10000} & {untargeted_SVMOPT_gdas_1000} & {untargeted_SVMOPT_gdas_2000} & {untargeted_SVMOPT_gdas_5000} & {untargeted_SVMOPT_gdas_8000} & {untargeted_SVMOPT_gdas_10000}  \\\\
                        & HSJA ($B_0=100$) & {targeted_HSJA_gdas_1000} & {targeted_HSJA_gdas_2000} & {targeted_HSJA_gdas_5000} & {targeted_HSJA_gdas_8000} & {targeted_HSJA_gdas_10000} & {untargeted_HSJA_gdas_1000} & {untargeted_HSJA_gdas_2000} & {untargeted_HSJA_gdas_5000} & {untargeted_HSJA_gdas_8000} & {untargeted_HSJA_gdas_10000}  \\\\
                        & Ours ($B_0=100$) & {targeted_Tangent_gdas_1000} & {targeted_Tangent_gdas_2000} & {targeted_Tangent_gdas_5000} & {targeted_Tangent_gdas_8000} & {targeted_Tangent_gdas_10000} & {untargeted_Tangent_gdas_1000} & {untargeted_Tangent_gdas_2000} & {untargeted_Tangent_gdas_5000} & {untargeted_Tangent_gdas_8000} & {untargeted_Tangent_gdas_10000}  \\\\
                        & HSJA ($B_0=30$) & {targeted_HSJA30_gdas_1000} & {targeted_HSJA30_gdas_2000} & {targeted_HSJA30_gdas_5000} & {targeted_HSJA30_gdas_8000} & {targeted_HSJA30_gdas_10000} & {untargeted_HSJA30_gdas_1000} & {untargeted_HSJA30_gdas_2000} & {untargeted_HSJA30_gdas_5000} & {untargeted_HSJA30_gdas_8000} & {untargeted_HSJA30_gdas_10000}  \\\\
                        & Ours ($B_0=30$) & {targeted_Tangent30_gdas_1000} & {targeted_Tangent30_gdas_2000} & {targeted_Tangent30_gdas_5000} & {targeted_Tangent30_gdas_8000} & {targeted_Tangent30_gdas_10000} & {untargeted_Tangent30_gdas_1000} & {untargeted_Tangent30_gdas_2000} & {untargeted_Tangent30_gdas_5000} & {untargeted_Tangent30_gdas_8000} & {untargeted_Tangent30_gdas_10000}  \\\\
                        \\midrule
                        & Boundary & {targeted_Boundary_WRN28_1000} & {targeted_Boundary_WRN28_2000} & {targeted_Boundary_WRN28_5000} & {targeted_Boundary_WRN28_8000} & {targeted_Boundary_WRN28_10000} & {untargeted_Boundary_WRN28_1000} & {untargeted_Boundary_WRN28_2000} & {untargeted_Boundary_WRN28_5000} & {untargeted_Boundary_WRN28_8000} & {untargeted_Boundary_WRN28_10000}  \\\\
                        & Sign-OPT & {targeted_SignOPT_WRN28_1000} & {targeted_SignOPT_WRN28_2000} & {targeted_SignOPT_WRN28_5000} & {targeted_SignOPT_WRN28_8000} & {targeted_SignOPT_WRN28_10000} & {untargeted_SignOPT_WRN28_1000} & {untargeted_SignOPT_WRN28_2000} & {untargeted_SignOPT_WRN28_5000} & {untargeted_SignOPT_WRN28_8000} & {untargeted_SignOPT_WRN28_10000}  \\\\
                        & SVM-OPT & {targeted_SVMOPT_WRN28_1000} & {targeted_SVMOPT_WRN28_2000} & {targeted_SVMOPT_WRN28_5000} & {targeted_SVMOPT_WRN28_8000} & {targeted_SVMOPT_WRN28_10000} & {untargeted_SVMOPT_WRN28_1000} & {untargeted_SVMOPT_WRN28_2000} & {untargeted_SVMOPT_WRN28_5000} & {untargeted_SVMOPT_WRN28_8000} & {untargeted_SVMOPT_WRN28_10000}  \\\\
                        & HSJA ($B_0=100$) & {targeted_HSJA_WRN28_1000} & {targeted_HSJA_WRN28_2000} & {targeted_HSJA_WRN28_5000} & {targeted_HSJA_WRN28_8000} & {targeted_HSJA_WRN28_10000} & {untargeted_HSJA_WRN28_1000} & {untargeted_HSJA_WRN28_2000} & {untargeted_HSJA_WRN28_5000} & {untargeted_HSJA_WRN28_8000} & {untargeted_HSJA_WRN28_10000}  \\\\
                        & Ours ($B_0=100$) & {targeted_Tangent_WRN28_1000} & {targeted_Tangent_WRN28_2000} & {targeted_Tangent_WRN28_5000} & {targeted_Tangent_WRN28_8000} & {targeted_Tangent_WRN28_10000} & {untargeted_Tangent_WRN28_1000} & {untargeted_Tangent_WRN28_2000} & {untargeted_Tangent_WRN28_5000} & {untargeted_Tangent_WRN28_8000} & {untargeted_Tangent_WRN28_10000}  \\\\
                        & HSJA ($B_0=30$) & {targeted_HSJA30_WRN28_1000} & {targeted_HSJA30_WRN28_2000} & {targeted_HSJA30_WRN28_5000} & {targeted_HSJA30_WRN28_8000} & {targeted_HSJA30_WRN28_10000} & {untargeted_HSJA30_WRN28_1000} & {untargeted_HSJA30_WRN28_2000} & {untargeted_HSJA30_WRN28_5000} & {untargeted_HSJA30_WRN28_8000} & {untargeted_HSJA30_WRN28_10000}  \\\\
                        & Ours ($B_0=30$) & {targeted_Tangent30_WRN28_1000} & {targeted_Tangent30_WRN28_2000} & {targeted_Tangent30_WRN28_5000} & {targeted_Tangent30_WRN28_8000} & {targeted_Tangent30_WRN28_10000} & {untargeted_Tangent30_WRN28_1000} & {untargeted_Tangent30_WRN28_2000} & {untargeted_Tangent30_WRN28_5000} & {untargeted_Tangent30_WRN28_8000} & {untargeted_Tangent30_WRN28_10000}  \\\\
                        \\midrule
                        & Boundary & {targeted_Boundary_WRN40_1000} & {targeted_Boundary_WRN40_2000} & {targeted_Boundary_WRN40_5000} & {targeted_Boundary_WRN40_8000} & {targeted_Boundary_WRN40_10000} & {untargeted_Boundary_WRN40_1000} & {untargeted_Boundary_WRN40_2000} & {untargeted_Boundary_WRN40_5000} & {untargeted_Boundary_WRN40_8000} & {untargeted_Boundary_WRN40_10000}  \\\\
                        & Sign-OPT & {targeted_SignOPT_WRN40_1000} & {targeted_SignOPT_WRN40_2000} & {targeted_SignOPT_WRN40_5000} & {targeted_SignOPT_WRN40_8000} & {targeted_SignOPT_WRN40_10000} & {untargeted_SignOPT_WRN40_1000} & {untargeted_SignOPT_WRN40_2000} & {untargeted_SignOPT_WRN40_5000} & {untargeted_SignOPT_WRN40_8000} & {untargeted_SignOPT_WRN40_10000}  \\\\
                        & SVM-OPT & {targeted_SVMOPT_WRN40_1000} & {targeted_SVMOPT_WRN40_2000} & {targeted_SVMOPT_WRN40_5000} & {targeted_SVMOPT_WRN40_8000} & {targeted_SVMOPT_WRN40_10000} & {untargeted_SVMOPT_WRN40_1000} & {untargeted_SVMOPT_WRN40_2000} & {untargeted_SVMOPT_WRN40_5000} & {untargeted_SVMOPT_WRN40_8000} & {untargeted_SVMOPT_WRN40_10000}  \\\\
                        & HSJA ($B_0=100$) & {targeted_HSJA_WRN40_1000} & {targeted_HSJA_WRN40_2000} & {targeted_HSJA_WRN40_5000} & {targeted_HSJA_WRN40_8000} & {targeted_HSJA_WRN40_10000} & {untargeted_HSJA_WRN40_1000} & {untargeted_HSJA_WRN40_2000} & {untargeted_HSJA_WRN40_5000} & {untargeted_HSJA_WRN40_8000} & {untargeted_HSJA_WRN40_10000}  \\\\
                        & Ours ($B_0=100$)& {targeted_Tangent_WRN40_1000} & {targeted_Tangent_WRN40_2000} & {targeted_Tangent_WRN40_5000} & {targeted_Tangent_WRN40_8000} & {targeted_Tangent_WRN40_10000} & {untargeted_Tangent_WRN40_1000} & {untargeted_Tangent_WRN40_2000} & {untargeted_Tangent_WRN40_5000} & {untargeted_Tangent_WRN40_8000} & {untargeted_Tangent_WRN40_10000}  \\\\
                        & HSJA ($B_0=30$) & {targeted_HSJA30_WRN40_1000} & {targeted_HSJA30_WRN40_2000} & {targeted_HSJA30_WRN40_5000} & {targeted_HSJA30_WRN40_8000} & {targeted_HSJA30_WRN40_10000} & {untargeted_HSJA30_WRN40_1000} & {untargeted_HSJA30_WRN40_2000} & {untargeted_HSJA30_WRN40_5000} & {untargeted_HSJA30_WRN40_8000} & {untargeted_HSJA30_WRN40_10000}  \\\\
                        & Ours ($B_0=30$) & {targeted_Tangent30_WRN40_1000} & {targeted_Tangent30_WRN40_2000} & {targeted_Tangent30_WRN40_5000} & {targeted_Tangent30_WRN40_8000} & {targeted_Tangent30_WRN40_10000} & {untargeted_Tangent30_WRN40_1000} & {untargeted_Tangent30_WRN40_2000} & {untargeted_Tangent30_WRN40_5000} & {untargeted_Tangent30_WRN40_8000} & {untargeted_Tangent30_WRN40_10000}  \\\\
                            """.format(

        untargeted_SignOPT_pyramidnet272_1000=untargeted_result["pyramidnet272"]["Sign-OPT"][1000],
        untargeted_SignOPT_pyramidnet272_2000=untargeted_result["pyramidnet272"]["Sign-OPT"][2000],
        untargeted_SignOPT_pyramidnet272_5000=untargeted_result["pyramidnet272"]["Sign-OPT"][5000],
        untargeted_SignOPT_pyramidnet272_8000=untargeted_result["pyramidnet272"]["Sign-OPT"][8000],
        untargeted_SignOPT_pyramidnet272_10000=untargeted_result["pyramidnet272"]["Sign-OPT"][10000],

        targeted_SignOPT_pyramidnet272_1000=targeted_result["pyramidnet272"]["Sign-OPT"][1000],
        targeted_SignOPT_pyramidnet272_2000=targeted_result["pyramidnet272"]["Sign-OPT"][2000],
        targeted_SignOPT_pyramidnet272_5000=targeted_result["pyramidnet272"]["Sign-OPT"][5000],
        targeted_SignOPT_pyramidnet272_8000=targeted_result["pyramidnet272"]["Sign-OPT"][8000],
        targeted_SignOPT_pyramidnet272_10000=targeted_result["pyramidnet272"]["Sign-OPT"][10000],

        untargeted_SignOPT_gdas_1000=untargeted_result["gdas"]["Sign-OPT"][1000],
        untargeted_SignOPT_gdas_2000=untargeted_result["gdas"]["Sign-OPT"][2000],
        untargeted_SignOPT_gdas_5000=untargeted_result["gdas"]["Sign-OPT"][5000],
        untargeted_SignOPT_gdas_8000=untargeted_result["gdas"]["Sign-OPT"][8000],
        untargeted_SignOPT_gdas_10000=untargeted_result["gdas"]["Sign-OPT"][10000],

        targeted_SignOPT_gdas_1000=targeted_result["gdas"]["Sign-OPT"][1000],
        targeted_SignOPT_gdas_2000=targeted_result["gdas"]["Sign-OPT"][2000],
        targeted_SignOPT_gdas_5000=targeted_result["gdas"]["Sign-OPT"][5000],
        targeted_SignOPT_gdas_8000=targeted_result["gdas"]["Sign-OPT"][8000],
        targeted_SignOPT_gdas_10000=targeted_result["gdas"]["Sign-OPT"][10000],

        untargeted_SignOPT_WRN28_1000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][1000],
        untargeted_SignOPT_WRN28_2000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][2000],
        untargeted_SignOPT_WRN28_5000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][5000],
        untargeted_SignOPT_WRN28_8000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][8000],
        untargeted_SignOPT_WRN28_10000=untargeted_result["WRN-28-10-drop"]["Sign-OPT"][10000],

        targeted_SignOPT_WRN28_1000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][1000],
        targeted_SignOPT_WRN28_2000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][2000],
        targeted_SignOPT_WRN28_5000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][5000],
        targeted_SignOPT_WRN28_8000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][8000],
        targeted_SignOPT_WRN28_10000=targeted_result["WRN-28-10-drop"]["Sign-OPT"][10000],

        untargeted_SignOPT_WRN40_1000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][1000],
        untargeted_SignOPT_WRN40_2000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][2000],
        untargeted_SignOPT_WRN40_5000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][5000],
        untargeted_SignOPT_WRN40_8000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][8000],
        untargeted_SignOPT_WRN40_10000=untargeted_result["WRN-40-10-drop"]["Sign-OPT"][10000],

        targeted_SignOPT_WRN40_1000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][1000],
        targeted_SignOPT_WRN40_2000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][2000],
        targeted_SignOPT_WRN40_5000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][5000],
        targeted_SignOPT_WRN40_8000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][8000],
        targeted_SignOPT_WRN40_10000=targeted_result["WRN-40-10-drop"]["Sign-OPT"][10000],

        untargeted_SVMOPT_pyramidnet272_1000=untargeted_result["pyramidnet272"]["SVM-OPT"][1000],
        untargeted_SVMOPT_pyramidnet272_2000=untargeted_result["pyramidnet272"]["SVM-OPT"][2000],
        untargeted_SVMOPT_pyramidnet272_5000=untargeted_result["pyramidnet272"]["SVM-OPT"][5000],
        untargeted_SVMOPT_pyramidnet272_8000=untargeted_result["pyramidnet272"]["SVM-OPT"][8000],
        untargeted_SVMOPT_pyramidnet272_10000=untargeted_result["pyramidnet272"]["SVM-OPT"][10000],

        targeted_SVMOPT_pyramidnet272_1000=targeted_result["pyramidnet272"]["SVM-OPT"][1000],
        targeted_SVMOPT_pyramidnet272_2000=targeted_result["pyramidnet272"]["SVM-OPT"][2000],
        targeted_SVMOPT_pyramidnet272_5000=targeted_result["pyramidnet272"]["SVM-OPT"][5000],
        targeted_SVMOPT_pyramidnet272_8000=targeted_result["pyramidnet272"]["SVM-OPT"][8000],
        targeted_SVMOPT_pyramidnet272_10000=targeted_result["pyramidnet272"]["SVM-OPT"][10000],

        untargeted_SVMOPT_gdas_1000=untargeted_result["gdas"]["SVM-OPT"][1000],
        untargeted_SVMOPT_gdas_2000=untargeted_result["gdas"]["SVM-OPT"][2000],
        untargeted_SVMOPT_gdas_5000=untargeted_result["gdas"]["SVM-OPT"][5000],
        untargeted_SVMOPT_gdas_8000=untargeted_result["gdas"]["SVM-OPT"][8000],
        untargeted_SVMOPT_gdas_10000=untargeted_result["gdas"]["SVM-OPT"][10000],

        targeted_SVMOPT_gdas_1000=targeted_result["gdas"]["SVM-OPT"][1000],
        targeted_SVMOPT_gdas_2000=targeted_result["gdas"]["SVM-OPT"][2000],
        targeted_SVMOPT_gdas_5000=targeted_result["gdas"]["SVM-OPT"][5000],
        targeted_SVMOPT_gdas_8000=targeted_result["gdas"]["SVM-OPT"][8000],
        targeted_SVMOPT_gdas_10000=targeted_result["gdas"]["SVM-OPT"][10000],

        untargeted_SVMOPT_WRN28_1000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][1000],
        untargeted_SVMOPT_WRN28_2000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][2000],
        untargeted_SVMOPT_WRN28_5000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][5000],
        untargeted_SVMOPT_WRN28_8000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][8000],
        untargeted_SVMOPT_WRN28_10000=untargeted_result["WRN-28-10-drop"]["SVM-OPT"][10000],

        targeted_SVMOPT_WRN28_1000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][1000],
        targeted_SVMOPT_WRN28_2000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][2000],
        targeted_SVMOPT_WRN28_5000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][5000],
        targeted_SVMOPT_WRN28_8000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][8000],
        targeted_SVMOPT_WRN28_10000=targeted_result["WRN-28-10-drop"]["SVM-OPT"][10000],

        untargeted_SVMOPT_WRN40_1000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][1000],
        untargeted_SVMOPT_WRN40_2000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][2000],
        untargeted_SVMOPT_WRN40_5000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][5000],
        untargeted_SVMOPT_WRN40_8000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][8000],
        untargeted_SVMOPT_WRN40_10000=untargeted_result["WRN-40-10-drop"]["SVM-OPT"][10000],

        targeted_SVMOPT_WRN40_1000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][1000],
        targeted_SVMOPT_WRN40_2000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][2000],
        targeted_SVMOPT_WRN40_5000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][5000],
        targeted_SVMOPT_WRN40_8000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][8000],
        targeted_SVMOPT_WRN40_10000=targeted_result["WRN-40-10-drop"]["SVM-OPT"][10000],

        untargeted_Boundary_pyramidnet272_1000=untargeted_result["pyramidnet272"]["Boundary Attack"][1000],
        untargeted_Boundary_pyramidnet272_2000=untargeted_result["pyramidnet272"]["Boundary Attack"][2000],
        untargeted_Boundary_pyramidnet272_5000=untargeted_result["pyramidnet272"]["Boundary Attack"][5000],
        untargeted_Boundary_pyramidnet272_8000=untargeted_result["pyramidnet272"]["Boundary Attack"][8000],
        untargeted_Boundary_pyramidnet272_10000=untargeted_result["pyramidnet272"]["Boundary Attack"][10000],

        targeted_Boundary_pyramidnet272_1000=targeted_result["pyramidnet272"]["Boundary Attack"][1000],
        targeted_Boundary_pyramidnet272_2000=targeted_result["pyramidnet272"]["Boundary Attack"][2000],
        targeted_Boundary_pyramidnet272_5000=targeted_result["pyramidnet272"]["Boundary Attack"][5000],
        targeted_Boundary_pyramidnet272_8000=targeted_result["pyramidnet272"]["Boundary Attack"][8000],
        targeted_Boundary_pyramidnet272_10000=targeted_result["pyramidnet272"]["Boundary Attack"][10000],

        untargeted_Boundary_gdas_1000=untargeted_result["gdas"]["Boundary Attack"][1000],
        untargeted_Boundary_gdas_2000=untargeted_result["gdas"]["Boundary Attack"][2000],
        untargeted_Boundary_gdas_5000=untargeted_result["gdas"]["Boundary Attack"][5000],
        untargeted_Boundary_gdas_8000=untargeted_result["gdas"]["Boundary Attack"][8000],
        untargeted_Boundary_gdas_10000=untargeted_result["gdas"]["Boundary Attack"][10000],

        targeted_Boundary_gdas_1000=targeted_result["gdas"]["Boundary Attack"][1000],
        targeted_Boundary_gdas_2000=targeted_result["gdas"]["Boundary Attack"][2000],
        targeted_Boundary_gdas_5000=targeted_result["gdas"]["Boundary Attack"][5000],
        targeted_Boundary_gdas_8000=targeted_result["gdas"]["Boundary Attack"][8000],
        targeted_Boundary_gdas_10000=targeted_result["gdas"]["Boundary Attack"][10000],

        untargeted_Boundary_WRN28_1000=untargeted_result["WRN-28-10-drop"]["Boundary Attack"][1000],
        untargeted_Boundary_WRN28_2000=untargeted_result["WRN-28-10-drop"]["Boundary Attack"][2000],
        untargeted_Boundary_WRN28_5000=untargeted_result["WRN-28-10-drop"]["Boundary Attack"][5000],
        untargeted_Boundary_WRN28_8000=untargeted_result["WRN-28-10-drop"]["Boundary Attack"][8000],
        untargeted_Boundary_WRN28_10000=untargeted_result["WRN-28-10-drop"]["Boundary Attack"][10000],

        targeted_Boundary_WRN28_1000=targeted_result["WRN-28-10-drop"]["Boundary Attack"][1000],
        targeted_Boundary_WRN28_2000=targeted_result["WRN-28-10-drop"]["Boundary Attack"][2000],
        targeted_Boundary_WRN28_5000=targeted_result["WRN-28-10-drop"]["Boundary Attack"][5000],
        targeted_Boundary_WRN28_8000=targeted_result["WRN-28-10-drop"]["Boundary Attack"][8000],
        targeted_Boundary_WRN28_10000=targeted_result["WRN-28-10-drop"]["Boundary Attack"][10000],

        untargeted_Boundary_WRN40_1000=untargeted_result["WRN-40-10-drop"]["Boundary Attack"][1000],
        untargeted_Boundary_WRN40_2000=untargeted_result["WRN-40-10-drop"]["Boundary Attack"][2000],
        untargeted_Boundary_WRN40_5000=untargeted_result["WRN-40-10-drop"]["Boundary Attack"][5000],
        untargeted_Boundary_WRN40_8000=untargeted_result["WRN-40-10-drop"]["Boundary Attack"][8000],
        untargeted_Boundary_WRN40_10000=untargeted_result["WRN-40-10-drop"]["Boundary Attack"][10000],

        targeted_Boundary_WRN40_1000=targeted_result["WRN-40-10-drop"]["Boundary Attack"][1000],
        targeted_Boundary_WRN40_2000=targeted_result["WRN-40-10-drop"]["Boundary Attack"][2000],
        targeted_Boundary_WRN40_5000=targeted_result["WRN-40-10-drop"]["Boundary Attack"][5000],
        targeted_Boundary_WRN40_8000=targeted_result["WRN-40-10-drop"]["Boundary Attack"][8000],
        targeted_Boundary_WRN40_10000=targeted_result["WRN-40-10-drop"]["Boundary Attack"][10000],

        untargeted_HSJA_pyramidnet272_1000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_pyramidnet272_2000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_pyramidnet272_5000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_pyramidnet272_8000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_pyramidnet272_10000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_pyramidnet272_1000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_pyramidnet272_2000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_pyramidnet272_5000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_pyramidnet272_8000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_pyramidnet272_10000=targeted_result["pyramidnet272"]["HopSkipJumpAttack"][10000],

        untargeted_HSJA_gdas_1000=untargeted_result["gdas"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_gdas_2000=untargeted_result["gdas"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_gdas_5000=untargeted_result["gdas"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_gdas_8000=untargeted_result["gdas"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_gdas_10000=untargeted_result["gdas"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_gdas_1000=targeted_result["gdas"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_gdas_2000=targeted_result["gdas"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_gdas_5000=targeted_result["gdas"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_gdas_8000=targeted_result["gdas"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_gdas_10000=targeted_result["gdas"]["HopSkipJumpAttack"][10000],

        untargeted_HSJA_WRN28_1000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_WRN28_2000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_WRN28_5000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_WRN28_8000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_WRN28_10000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_WRN28_1000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_WRN28_2000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_WRN28_5000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_WRN28_8000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_WRN28_10000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack"][10000],

        untargeted_HSJA_WRN40_1000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][1000],
        untargeted_HSJA_WRN40_2000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][2000],
        untargeted_HSJA_WRN40_5000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][5000],
        untargeted_HSJA_WRN40_8000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][8000],
        untargeted_HSJA_WRN40_10000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][10000],

        targeted_HSJA_WRN40_1000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][1000],
        targeted_HSJA_WRN40_2000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][2000],
        targeted_HSJA_WRN40_5000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][5000],
        targeted_HSJA_WRN40_8000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][8000],
        targeted_HSJA_WRN40_10000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack"][10000],

        untargeted_Tangent_pyramidnet272_1000=untargeted_result["pyramidnet272"]["Tangent Attack"][1000],
        untargeted_Tangent_pyramidnet272_2000=untargeted_result["pyramidnet272"]["Tangent Attack"][2000],
        untargeted_Tangent_pyramidnet272_5000=untargeted_result["pyramidnet272"]["Tangent Attack"][5000],
        untargeted_Tangent_pyramidnet272_8000=untargeted_result["pyramidnet272"]["Tangent Attack"][8000],
        untargeted_Tangent_pyramidnet272_10000=untargeted_result["pyramidnet272"]["Tangent Attack"][10000],

        targeted_Tangent_pyramidnet272_1000=targeted_result["pyramidnet272"]["Tangent Attack"][1000],
        targeted_Tangent_pyramidnet272_2000=targeted_result["pyramidnet272"]["Tangent Attack"][2000],
        targeted_Tangent_pyramidnet272_5000=targeted_result["pyramidnet272"]["Tangent Attack"][5000],
        targeted_Tangent_pyramidnet272_8000=targeted_result["pyramidnet272"]["Tangent Attack"][8000],
        targeted_Tangent_pyramidnet272_10000=targeted_result["pyramidnet272"]["Tangent Attack"][10000],

        untargeted_Tangent_gdas_1000=untargeted_result["gdas"]["Tangent Attack"][1000],
        untargeted_Tangent_gdas_2000=untargeted_result["gdas"]["Tangent Attack"][2000],
        untargeted_Tangent_gdas_5000=untargeted_result["gdas"]["Tangent Attack"][5000],
        untargeted_Tangent_gdas_8000=untargeted_result["gdas"]["Tangent Attack"][8000],
        untargeted_Tangent_gdas_10000=untargeted_result["gdas"]["Tangent Attack"][10000],

        targeted_Tangent_gdas_1000=targeted_result["gdas"]["Tangent Attack"][1000],
        targeted_Tangent_gdas_2000=targeted_result["gdas"]["Tangent Attack"][2000],
        targeted_Tangent_gdas_5000=targeted_result["gdas"]["Tangent Attack"][5000],
        targeted_Tangent_gdas_8000=targeted_result["gdas"]["Tangent Attack"][8000],
        targeted_Tangent_gdas_10000=targeted_result["gdas"]["Tangent Attack"][10000],

        untargeted_Tangent_WRN28_1000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][1000],
        untargeted_Tangent_WRN28_2000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][2000],
        untargeted_Tangent_WRN28_5000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][5000],
        untargeted_Tangent_WRN28_8000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][8000],
        untargeted_Tangent_WRN28_10000=untargeted_result["WRN-28-10-drop"]["Tangent Attack"][10000],

        targeted_Tangent_WRN28_1000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][1000],
        targeted_Tangent_WRN28_2000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][2000],
        targeted_Tangent_WRN28_5000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][5000],
        targeted_Tangent_WRN28_8000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][8000],
        targeted_Tangent_WRN28_10000=targeted_result["WRN-28-10-drop"]["Tangent Attack"][10000],

        untargeted_Tangent_WRN40_1000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][1000],
        untargeted_Tangent_WRN40_2000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][2000],
        untargeted_Tangent_WRN40_5000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][5000],
        untargeted_Tangent_WRN40_8000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][8000],
        untargeted_Tangent_WRN40_10000=untargeted_result["WRN-40-10-drop"]["Tangent Attack"][10000],

        targeted_Tangent_WRN40_1000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][1000],
        targeted_Tangent_WRN40_2000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][2000],
        targeted_Tangent_WRN40_5000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][5000],
        targeted_Tangent_WRN40_8000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][8000],
        targeted_Tangent_WRN40_10000=targeted_result["WRN-40-10-drop"]["Tangent Attack"][10000],

        untargeted_HSJA30_pyramidnet272_1000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack30"][1000],
        untargeted_HSJA30_pyramidnet272_2000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack30"][2000],
        untargeted_HSJA30_pyramidnet272_5000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack30"][5000],
        untargeted_HSJA30_pyramidnet272_8000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack30"][8000],
        untargeted_HSJA30_pyramidnet272_10000=untargeted_result["pyramidnet272"]["HopSkipJumpAttack30"][10000],

        targeted_HSJA30_pyramidnet272_1000=targeted_result["pyramidnet272"]["HopSkipJumpAttack30"][1000],
        targeted_HSJA30_pyramidnet272_2000=targeted_result["pyramidnet272"]["HopSkipJumpAttack30"][2000],
        targeted_HSJA30_pyramidnet272_5000=targeted_result["pyramidnet272"]["HopSkipJumpAttack30"][5000],
        targeted_HSJA30_pyramidnet272_8000=targeted_result["pyramidnet272"]["HopSkipJumpAttack30"][8000],
        targeted_HSJA30_pyramidnet272_10000=targeted_result["pyramidnet272"]["HopSkipJumpAttack30"][10000],

        untargeted_HSJA30_gdas_1000=untargeted_result["gdas"]["HopSkipJumpAttack30"][1000],
        untargeted_HSJA30_gdas_2000=untargeted_result["gdas"]["HopSkipJumpAttack30"][2000],
        untargeted_HSJA30_gdas_5000=untargeted_result["gdas"]["HopSkipJumpAttack30"][5000],
        untargeted_HSJA30_gdas_8000=untargeted_result["gdas"]["HopSkipJumpAttack30"][8000],
        untargeted_HSJA30_gdas_10000=untargeted_result["gdas"]["HopSkipJumpAttack30"][10000],

        targeted_HSJA30_gdas_1000=targeted_result["gdas"]["HopSkipJumpAttack30"][1000],
        targeted_HSJA30_gdas_2000=targeted_result["gdas"]["HopSkipJumpAttack30"][2000],
        targeted_HSJA30_gdas_5000=targeted_result["gdas"]["HopSkipJumpAttack30"][5000],
        targeted_HSJA30_gdas_8000=targeted_result["gdas"]["HopSkipJumpAttack30"][8000],
        targeted_HSJA30_gdas_10000=targeted_result["gdas"]["HopSkipJumpAttack30"][10000],

        untargeted_HSJA30_WRN28_1000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack30"][1000],
        untargeted_HSJA30_WRN28_2000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack30"][2000],
        untargeted_HSJA30_WRN28_5000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack30"][5000],
        untargeted_HSJA30_WRN28_8000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack30"][8000],
        untargeted_HSJA30_WRN28_10000=untargeted_result["WRN-28-10-drop"]["HopSkipJumpAttack30"][10000],

        targeted_HSJA30_WRN28_1000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack30"][1000],
        targeted_HSJA30_WRN28_2000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack30"][2000],
        targeted_HSJA30_WRN28_5000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack30"][5000],
        targeted_HSJA30_WRN28_8000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack30"][8000],
        targeted_HSJA30_WRN28_10000=targeted_result["WRN-28-10-drop"]["HopSkipJumpAttack30"][10000],

        untargeted_HSJA30_WRN40_1000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack30"][1000],
        untargeted_HSJA30_WRN40_2000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack30"][2000],
        untargeted_HSJA30_WRN40_5000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack30"][5000],
        untargeted_HSJA30_WRN40_8000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack30"][8000],
        untargeted_HSJA30_WRN40_10000=untargeted_result["WRN-40-10-drop"]["HopSkipJumpAttack30"][10000],

        targeted_HSJA30_WRN40_1000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack30"][1000],
        targeted_HSJA30_WRN40_2000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack30"][2000],
        targeted_HSJA30_WRN40_5000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack30"][5000],
        targeted_HSJA30_WRN40_8000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack30"][8000],
        targeted_HSJA30_WRN40_10000=targeted_result["WRN-40-10-drop"]["HopSkipJumpAttack30"][10000],

        untargeted_Tangent30_pyramidnet272_1000=untargeted_result["pyramidnet272"]["Tangent Attack30"][1000],
        untargeted_Tangent30_pyramidnet272_2000=untargeted_result["pyramidnet272"]["Tangent Attack30"][2000],
        untargeted_Tangent30_pyramidnet272_5000=untargeted_result["pyramidnet272"]["Tangent Attack30"][5000],
        untargeted_Tangent30_pyramidnet272_8000=untargeted_result["pyramidnet272"]["Tangent Attack30"][8000],
        untargeted_Tangent30_pyramidnet272_10000=untargeted_result["pyramidnet272"]["Tangent Attack30"][10000],

        targeted_Tangent30_pyramidnet272_1000=targeted_result["pyramidnet272"]["Tangent Attack30"][1000],
        targeted_Tangent30_pyramidnet272_2000=targeted_result["pyramidnet272"]["Tangent Attack30"][2000],
        targeted_Tangent30_pyramidnet272_5000=targeted_result["pyramidnet272"]["Tangent Attack30"][5000],
        targeted_Tangent30_pyramidnet272_8000=targeted_result["pyramidnet272"]["Tangent Attack30"][8000],
        targeted_Tangent30_pyramidnet272_10000=targeted_result["pyramidnet272"]["Tangent Attack30"][10000],

        untargeted_Tangent30_gdas_1000=untargeted_result["gdas"]["Tangent Attack30"][1000],
        untargeted_Tangent30_gdas_2000=untargeted_result["gdas"]["Tangent Attack30"][2000],
        untargeted_Tangent30_gdas_5000=untargeted_result["gdas"]["Tangent Attack30"][5000],
        untargeted_Tangent30_gdas_8000=untargeted_result["gdas"]["Tangent Attack30"][8000],
        untargeted_Tangent30_gdas_10000=untargeted_result["gdas"]["Tangent Attack30"][10000],

        targeted_Tangent30_gdas_1000=targeted_result["gdas"]["Tangent Attack30"][1000],
        targeted_Tangent30_gdas_2000=targeted_result["gdas"]["Tangent Attack30"][2000],
        targeted_Tangent30_gdas_5000=targeted_result["gdas"]["Tangent Attack30"][5000],
        targeted_Tangent30_gdas_8000=targeted_result["gdas"]["Tangent Attack30"][8000],
        targeted_Tangent30_gdas_10000=targeted_result["gdas"]["Tangent Attack30"][10000],

        untargeted_Tangent30_WRN28_1000=untargeted_result["WRN-28-10-drop"]["Tangent Attack30"][1000],
        untargeted_Tangent30_WRN28_2000=untargeted_result["WRN-28-10-drop"]["Tangent Attack30"][2000],
        untargeted_Tangent30_WRN28_5000=untargeted_result["WRN-28-10-drop"]["Tangent Attack30"][5000],
        untargeted_Tangent30_WRN28_8000=untargeted_result["WRN-28-10-drop"]["Tangent Attack30"][8000],
        untargeted_Tangent30_WRN28_10000=untargeted_result["WRN-28-10-drop"]["Tangent Attack30"][10000],

        targeted_Tangent30_WRN28_1000=targeted_result["WRN-28-10-drop"]["Tangent Attack30"][1000],
        targeted_Tangent30_WRN28_2000=targeted_result["WRN-28-10-drop"]["Tangent Attack30"][2000],
        targeted_Tangent30_WRN28_5000=targeted_result["WRN-28-10-drop"]["Tangent Attack30"][5000],
        targeted_Tangent30_WRN28_8000=targeted_result["WRN-28-10-drop"]["Tangent Attack30"][8000],
        targeted_Tangent30_WRN28_10000=targeted_result["WRN-28-10-drop"]["Tangent Attack30"][10000],

        untargeted_Tangent30_WRN40_1000=untargeted_result["WRN-40-10-drop"]["Tangent Attack30"][1000],
        untargeted_Tangent30_WRN40_2000=untargeted_result["WRN-40-10-drop"]["Tangent Attack30"][2000],
        untargeted_Tangent30_WRN40_5000=untargeted_result["WRN-40-10-drop"]["Tangent Attack30"][5000],
        untargeted_Tangent30_WRN40_8000=untargeted_result["WRN-40-10-drop"]["Tangent Attack30"][8000],
        untargeted_Tangent30_WRN40_10000=untargeted_result["WRN-40-10-drop"]["Tangent Attack30"][10000],

        targeted_Tangent30_WRN40_1000=targeted_result["WRN-40-10-drop"]["Tangent Attack30"][1000],
        targeted_Tangent30_WRN40_2000=targeted_result["WRN-40-10-drop"]["Tangent Attack30"][2000],
        targeted_Tangent30_WRN40_5000=targeted_result["WRN-40-10-drop"]["Tangent Attack30"][5000],
        targeted_Tangent30_WRN40_8000=targeted_result["WRN-40-10-drop"]["Tangent Attack30"][8000],
        targeted_Tangent30_WRN40_10000=targeted_result["WRN-40-10-drop"]["Tangent Attack30"][10000],

    )
    )


def draw_tables_for_untargeted_CIFAR_without_BBA(archs_result):
    result = archs_result
    print("""
                    & & Sign-OPT & {SignOPT_pyramidnet272_1000} & {SignOPT_pyramidnet272_2000} & {SignOPT_pyramidnet272_5000} & {SignOPT_pyramidnet272_8000} & {SignOPT_pyramidnet272_10000} & - & - & - & - \\\\
                    & & SVM-OPT & {SVMOPT_pyramidnet272_1000} & {SVMOPT_pyramidnet272_2000} & {SVMOPT_pyramidnet272_5000} & {SVMOPT_pyramidnet272_8000} & {SVMOPT_pyramidnet272_10000} & - & - & - & - \\\\
                    & & GeoDA & {GeoDA_pyramidnet272_1000} & {GeoDA_pyramidnet272_2000} & {GeoDA_pyramidnet272_5000} & {GeoDA_pyramidnet272_8000} & {GeoDA_pyramidnet272_10000} & - & - & - & - \\\\
                    & & RayS & {RayS_pyramidnet272_1000} & {RayS_pyramidnet272_2000} & {RayS_pyramidnet272_5000} & {RayS_pyramidnet272_8000} & {RayS_pyramidnet272_10000} & - & - & - & - \\\\
                    & & HopSkipJumpAttack & {HSJA_pyramidnet272_1000} & {HSJA_pyramidnet272_2000} & {HSJA_pyramidnet272_5000} & {HSJA_pyramidnet272_8000} & {HSJA_pyramidnet272_10000} & - & - & - & - \\\\
                    & & Tangent Attack (ours) & {Tangent_pyramidnet272_1000} & {Tangent_pyramidnet272_2000} & {Tangent_pyramidnet272_5000} & {Tangent_pyramidnet272_8000} & {Tangent_pyramidnet272_10000} & - & - & - & - \\\\
                    \\cmidrule(rl){{2-12}}
                    & & Sign-OPT & {SignOPT_gdas_1000} & {SignOPT_gdas_2000} & {SignOPT_gdas_5000} & {SignOPT_gdas_8000} & {SignOPT_gdas_10000} & - & - & - & - \\\\
                    & & SVM-OPT & {SVMOPT_gdas_1000} & {SVMOPT_gdas_2000} & {SVMOPT_gdas_5000} & {SVMOPT_gdas_8000} & {SVMOPT_gdas_10000} & - & - & - & - \\\\
                    & & GeoDA & {GeoDA_gdas_1000} & {GeoDA_gdas_2000} & {GeoDA_gdas_5000} & {GeoDA_gdas_8000} & {GeoDA_gdas_10000} & - & - & - & - \\\\
                    & & RayS & {RayS_gdas_1000} & {RayS_gdas_2000} & {RayS_gdas_5000} & {RayS_gdas_8000} & {RayS_gdas_10000} & - & - & - & - \\\\
                    & & HopSkipJumpAttack & {HSJA_gdas_1000} & {HSJA_gdas_2000} & {HSJA_gdas_5000} & {HSJA_gdas_8000} & {HSJA_gdas_10000} & - & - & - & - \\\\
                    & & Tangent Attack (ours) & {Tangent_gdas_1000} & {Tangent_gdas_2000} & {Tangent_gdas_5000} & {Tangent_gdas_8000} & {Tangent_gdas_10000} & - & - & - & - \\\\
                    \\cmidrule(rl){{2-12}}
                    & & Sign-OPT & {SignOPT_WRN28_1000} & {SignOPT_WRN28_2000} & {SignOPT_WRN28_5000} & {SignOPT_WRN28_8000} & {SignOPT_WRN28_10000} & - & - & - & - \\\\
                    & & SVM-OPT & {SVMOPT_WRN28_1000} & {SVMOPT_WRN28_2000} & {SVMOPT_WRN28_5000} & {SVMOPT_WRN28_8000} & {SVMOPT_WRN28_10000} & - & - & - & - \\\\
                    & & GeoDA & {GeoDA_WRN28_1000} & {GeoDA_WRN28_2000} & {GeoDA_WRN28_5000} & {GeoDA_WRN28_8000} & {GeoDA_WRN28_10000} & - & - & - & - \\\\
                    & & RayS & {RayS_WRN28_1000} & {RayS_WRN28_2000} & {RayS_WRN28_5000} & {RayS_WRN28_8000} & {RayS_WRN28_10000} & - & - & - & - \\\\
                    & & HopSkipJumpAttack & {HSJA_WRN28_1000} & {HSJA_WRN28_2000} & {HSJA_WRN28_5000} & {HSJA_WRN28_8000} & {HSJA_WRN28_10000} & - & - & - & - \\\\
                    & & Tangent Attack (ours) & {Tangent_WRN28_1000} & {Tangent_WRN28_2000} & {Tangent_WRN28_5000} & {Tangent_WRN28_8000} & {Tangent_WRN28_10000} & - & - & - & - \\\\
                    \\cmidrule(rl){{2-12}}
                    & & Sign-OPT & {SignOPT_WRN40_1000} & {SignOPT_WRN40_2000} & {SignOPT_WRN40_5000} & {SignOPT_WRN40_8000} & {SignOPT_WRN40_10000} & - & - & - & - \\\\
                    & & SVM-OPT & {SVMOPT_WRN40_1000} & {SVMOPT_WRN40_2000} & {SVMOPT_WRN40_5000} & {SVMOPT_WRN40_8000} & {SVMOPT_WRN40_10000} & - & - & - & - \\\\
                    & & GeoDA & {GeoDA_WRN40_1000} & {GeoDA_WRN40_2000} & {GeoDA_WRN40_5000} & {GeoDA_WRN40_8000} & {GeoDA_WRN40_10000} & - & - & - & - \\\\
                    & & RayS & {RayS_WRN40_1000} & {RayS_WRN40_2000} & {RayS_WRN40_5000} & {RayS_WRN40_8000} & {RayS_WRN40_10000} & - & - & - & - \\\\
                    & & HopSkipJumpAttack & {HSJA_WRN40_1000} & {HSJA_WRN40_2000} & {HSJA_WRN40_5000} & {HSJA_WRN40_8000} & {HSJA_WRN40_10000} & - & - & - & - \\\\
                    & & Tangent Attack (ours) & {Tangent_WRN40_1000} & {Tangent_WRN40_2000} & {Tangent_WRN40_5000} & {Tangent_WRN40_8000} & {Tangent_WRN40_10000} & - & - & - & - \\\\
                    \\cmidrule(rl){{2-12}}
                    & & Sign-OPT & {SignOPT_densenet190_1000} & {SignOPT_densenet190_2000} & {SignOPT_densenet190_5000} & {SignOPT_densenet190_8000} & {SignOPT_densenet190_10000} & - & - & - & - \\\\
                    & & SVM-OPT & {SVMOPT_densenet190_1000} & {SVMOPT_densenet190_2000} & {SVMOPT_densenet190_5000} & {SVMOPT_densenet190_8000} & {SVMOPT_densenet190_10000} & - & - & - & - \\\\
                    & & GeoDA & {GeoDA_densenet190_1000} & {GeoDA_densenet190_2000} & {GeoDA_densenet190_5000} & {GeoDA_densenet190_8000} & {GeoDA_densenet190_10000} & - & - & - & - \\\\
                    & & RayS & {RayS_densenet190_1000} & {RayS_densenet190_2000} & {RayS_densenet190_5000} & {RayS_densenet190_8000} & {RayS_densenet190_10000} & - & - & - & - \\\\
                    & & HopSkipJumpAttack & {HSJA_densenet190_1000} & {HSJA_densenet190_2000} & {HSJA_densenet190_5000} & {HSJA_densenet190_8000} & {HSJA_densenet190_10000} & - & - & - & - \\\\
                    & & Tangent Attack (ours) & {Tangent_densenet190_1000} & {Tangent_densenet190_2000} & {Tangent_densenet190_5000} & {Tangent_densenet190_8000} & {Tangent_densenet190_10000} & - & - & - & - \\\\
                        """.format(

        SignOPT_pyramidnet272_1000=result["pyramidnet272"]["Sign-OPT"][1000],
        SignOPT_pyramidnet272_2000=result["pyramidnet272"]["Sign-OPT"][2000],
        SignOPT_pyramidnet272_5000=result["pyramidnet272"]["Sign-OPT"][5000],
        SignOPT_pyramidnet272_8000=result["pyramidnet272"]["Sign-OPT"][8000],
        SignOPT_pyramidnet272_10000=result["pyramidnet272"]["Sign-OPT"][10000],

        SignOPT_gdas_1000=result["gdas"]["Sign-OPT"][1000],
        SignOPT_gdas_2000=result["gdas"]["Sign-OPT"][2000],
        SignOPT_gdas_5000=result["gdas"]["Sign-OPT"][5000],
        SignOPT_gdas_8000=result["gdas"]["Sign-OPT"][8000],
        SignOPT_gdas_10000=result["gdas"]["Sign-OPT"][10000],

        SignOPT_WRN28_1000=result["WRN-28-10-drop"]["Sign-OPT"][1000],
        SignOPT_WRN28_2000=result["WRN-28-10-drop"]["Sign-OPT"][2000],
        SignOPT_WRN28_5000=result["WRN-28-10-drop"]["Sign-OPT"][5000],
        SignOPT_WRN28_8000=result["WRN-28-10-drop"]["Sign-OPT"][8000],
        SignOPT_WRN28_10000=result["WRN-28-10-drop"]["Sign-OPT"][10000],

        SignOPT_WRN40_1000=result["WRN-40-10-drop"]["Sign-OPT"][1000],
        SignOPT_WRN40_2000=result["WRN-40-10-drop"]["Sign-OPT"][2000],
        SignOPT_WRN40_5000=result["WRN-40-10-drop"]["Sign-OPT"][5000],
        SignOPT_WRN40_8000=result["WRN-40-10-drop"]["Sign-OPT"][8000],
        SignOPT_WRN40_10000=result["WRN-40-10-drop"]["Sign-OPT"][10000],

        SignOPT_densenet190_1000=result["densenet-bc-L190-k40"]["Sign-OPT"][1000],
        SignOPT_densenet190_2000=result["densenet-bc-L190-k40"]["Sign-OPT"][2000],
        SignOPT_densenet190_5000=result["densenet-bc-L190-k40"]["Sign-OPT"][5000],
        SignOPT_densenet190_8000=result["densenet-bc-L190-k40"]["Sign-OPT"][8000],
        SignOPT_densenet190_10000=result["densenet-bc-L190-k40"]["Sign-OPT"][10000],

        SVMOPT_pyramidnet272_1000=result["pyramidnet272"]["SVM-OPT"][1000],
        SVMOPT_pyramidnet272_2000=result["pyramidnet272"]["SVM-OPT"][2000],
        SVMOPT_pyramidnet272_5000=result["pyramidnet272"]["SVM-OPT"][5000],
        SVMOPT_pyramidnet272_8000=result["pyramidnet272"]["SVM-OPT"][8000],
        SVMOPT_pyramidnet272_10000=result["pyramidnet272"]["SVM-OPT"][10000],

        SVMOPT_gdas_1000=result["gdas"]["SVM-OPT"][1000],
        SVMOPT_gdas_2000=result["gdas"]["SVM-OPT"][2000],
        SVMOPT_gdas_5000=result["gdas"]["SVM-OPT"][5000],
        SVMOPT_gdas_8000=result["gdas"]["SVM-OPT"][8000],
        SVMOPT_gdas_10000=result["gdas"]["SVM-OPT"][10000],

        SVMOPT_WRN28_1000=result["WRN-28-10-drop"]["SVM-OPT"][1000],
        SVMOPT_WRN28_2000=result["WRN-28-10-drop"]["SVM-OPT"][2000],
        SVMOPT_WRN28_5000=result["WRN-28-10-drop"]["SVM-OPT"][5000],
        SVMOPT_WRN28_8000=result["WRN-28-10-drop"]["SVM-OPT"][8000],
        SVMOPT_WRN28_10000=result["WRN-28-10-drop"]["SVM-OPT"][10000],

        SVMOPT_WRN40_1000=result["WRN-40-10-drop"]["SVM-OPT"][1000],
        SVMOPT_WRN40_2000=result["WRN-40-10-drop"]["SVM-OPT"][2000],
        SVMOPT_WRN40_5000=result["WRN-40-10-drop"]["SVM-OPT"][5000],
        SVMOPT_WRN40_8000=result["WRN-40-10-drop"]["SVM-OPT"][8000],
        SVMOPT_WRN40_10000=result["WRN-40-10-drop"]["SVM-OPT"][10000],

        SVMOPT_densenet190_1000=result["densenet-bc-L190-k40"]["SVM-OPT"][1000],
        SVMOPT_densenet190_2000=result["densenet-bc-L190-k40"]["SVM-OPT"][2000],
        SVMOPT_densenet190_5000=result["densenet-bc-L190-k40"]["SVM-OPT"][5000],
        SVMOPT_densenet190_8000=result["densenet-bc-L190-k40"]["SVM-OPT"][8000],
        SVMOPT_densenet190_10000=result["densenet-bc-L190-k40"]["SVM-OPT"][10000],

        GeoDA_pyramidnet272_1000=result["pyramidnet272"]["GeoDA"][1000],
        GeoDA_pyramidnet272_2000=result["pyramidnet272"]["GeoDA"][2000],
        GeoDA_pyramidnet272_5000=result["pyramidnet272"]["GeoDA"][5000],
        GeoDA_pyramidnet272_8000=result["pyramidnet272"]["GeoDA"][8000],
        GeoDA_pyramidnet272_10000=result["pyramidnet272"]["GeoDA"][10000],

        GeoDA_gdas_1000=result["gdas"]["GeoDA"][1000],
        GeoDA_gdas_2000=result["gdas"]["GeoDA"][2000],
        GeoDA_gdas_5000=result["gdas"]["GeoDA"][5000],
        GeoDA_gdas_8000=result["gdas"]["GeoDA"][8000],
        GeoDA_gdas_10000=result["gdas"]["GeoDA"][10000],

        GeoDA_WRN28_1000=result["WRN-28-10-drop"]["GeoDA"][1000],
        GeoDA_WRN28_2000=result["WRN-28-10-drop"]["GeoDA"][2000],
        GeoDA_WRN28_5000=result["WRN-28-10-drop"]["GeoDA"][5000],
        GeoDA_WRN28_8000=result["WRN-28-10-drop"]["GeoDA"][8000],
        GeoDA_WRN28_10000=result["WRN-28-10-drop"]["GeoDA"][10000],

        GeoDA_WRN40_1000=result["WRN-40-10-drop"]["GeoDA"][1000],
        GeoDA_WRN40_2000=result["WRN-40-10-drop"]["GeoDA"][2000],
        GeoDA_WRN40_5000=result["WRN-40-10-drop"]["GeoDA"][5000],
        GeoDA_WRN40_8000=result["WRN-40-10-drop"]["GeoDA"][8000],
        GeoDA_WRN40_10000=result["WRN-40-10-drop"]["GeoDA"][10000],

        GeoDA_densenet190_1000=result["densenet-bc-L190-k40"]["GeoDA"][1000],
        GeoDA_densenet190_2000=result["densenet-bc-L190-k40"]["GeoDA"][2000],
        GeoDA_densenet190_5000=result["densenet-bc-L190-k40"]["GeoDA"][5000],
        GeoDA_densenet190_8000=result["densenet-bc-L190-k40"]["GeoDA"][8000],
        GeoDA_densenet190_10000=result["densenet-bc-L190-k40"]["GeoDA"][10000],

        RayS_pyramidnet272_1000=result["pyramidnet272"]["RayS"][1000],
        RayS_pyramidnet272_2000=result["pyramidnet272"]["RayS"][2000],
        RayS_pyramidnet272_5000=result["pyramidnet272"]["RayS"][5000],
        RayS_pyramidnet272_8000=result["pyramidnet272"]["RayS"][8000],
        RayS_pyramidnet272_10000=result["pyramidnet272"]["RayS"][10000],

        RayS_gdas_1000=result["gdas"]["RayS"][1000],
        RayS_gdas_2000=result["gdas"]["RayS"][2000],
        RayS_gdas_5000=result["gdas"]["RayS"][5000],
        RayS_gdas_8000=result["gdas"]["RayS"][8000],
        RayS_gdas_10000=result["gdas"]["RayS"][10000],

        RayS_WRN28_1000=result["WRN-28-10-drop"]["RayS"][1000],
        RayS_WRN28_2000=result["WRN-28-10-drop"]["RayS"][2000],
        RayS_WRN28_5000=result["WRN-28-10-drop"]["RayS"][5000],
        RayS_WRN28_8000=result["WRN-28-10-drop"]["RayS"][8000],
        RayS_WRN28_10000=result["WRN-28-10-drop"]["RayS"][10000],

        RayS_WRN40_1000=result["WRN-40-10-drop"]["RayS"][1000],
        RayS_WRN40_2000=result["WRN-40-10-drop"]["RayS"][2000],
        RayS_WRN40_5000=result["WRN-40-10-drop"]["RayS"][5000],
        RayS_WRN40_8000=result["WRN-40-10-drop"]["RayS"][8000],
        RayS_WRN40_10000=result["WRN-40-10-drop"]["RayS"][10000],

        RayS_densenet190_1000=result["densenet-bc-L190-k40"]["RayS"][1000],
        RayS_densenet190_2000=result["densenet-bc-L190-k40"]["RayS"][2000],
        RayS_densenet190_5000=result["densenet-bc-L190-k40"]["RayS"][5000],
        RayS_densenet190_8000=result["densenet-bc-L190-k40"]["RayS"][8000],
        RayS_densenet190_10000=result["densenet-bc-L190-k40"]["RayS"][10000],

        HSJA_pyramidnet272_1000=result["pyramidnet272"]["HopSkipJumpAttack"][1000],
        HSJA_pyramidnet272_2000=result["pyramidnet272"]["HopSkipJumpAttack"][2000],
        HSJA_pyramidnet272_5000=result["pyramidnet272"]["HopSkipJumpAttack"][5000],
        HSJA_pyramidnet272_8000=result["pyramidnet272"]["HopSkipJumpAttack"][8000],
        HSJA_pyramidnet272_10000=result["pyramidnet272"]["HopSkipJumpAttack"][10000],

        HSJA_gdas_1000=result["gdas"]["HopSkipJumpAttack"][1000],
        HSJA_gdas_2000=result["gdas"]["HopSkipJumpAttack"][2000],
        HSJA_gdas_5000=result["gdas"]["HopSkipJumpAttack"][5000],
        HSJA_gdas_8000=result["gdas"]["HopSkipJumpAttack"][8000],
        HSJA_gdas_10000=result["gdas"]["HopSkipJumpAttack"][10000],

        HSJA_WRN28_1000=result["WRN-28-10-drop"]["HopSkipJumpAttack"][1000],
        HSJA_WRN28_2000=result["WRN-28-10-drop"]["HopSkipJumpAttack"][2000],
        HSJA_WRN28_5000=result["WRN-28-10-drop"]["HopSkipJumpAttack"][5000],
        HSJA_WRN28_8000=result["WRN-28-10-drop"]["HopSkipJumpAttack"][8000],
        HSJA_WRN28_10000=result["WRN-28-10-drop"]["HopSkipJumpAttack"][10000],

        HSJA_WRN40_1000=result["WRN-40-10-drop"]["HopSkipJumpAttack"][1000],
        HSJA_WRN40_2000=result["WRN-40-10-drop"]["HopSkipJumpAttack"][2000],
        HSJA_WRN40_5000=result["WRN-40-10-drop"]["HopSkipJumpAttack"][5000],
        HSJA_WRN40_8000=result["WRN-40-10-drop"]["HopSkipJumpAttack"][8000],
        HSJA_WRN40_10000=result["WRN-40-10-drop"]["HopSkipJumpAttack"][10000],

        HSJA_densenet190_1000=result["densenet-bc-L190-k40"]["HopSkipJumpAttack"][1000],
        HSJA_densenet190_2000=result["densenet-bc-L190-k40"]["HopSkipJumpAttack"][2000],
        HSJA_densenet190_5000=result["densenet-bc-L190-k40"]["HopSkipJumpAttack"][5000],
        HSJA_densenet190_8000=result["densenet-bc-L190-k40"]["HopSkipJumpAttack"][8000],
        HSJA_densenet190_10000=result["densenet-bc-L190-k40"]["HopSkipJumpAttack"][10000],

        Tangent_pyramidnet272_1000=result["pyramidnet272"]["Tangent Attack"][1000],
        Tangent_pyramidnet272_2000=result["pyramidnet272"]["Tangent Attack"][2000],
        Tangent_pyramidnet272_5000=result["pyramidnet272"]["Tangent Attack"][5000],
        Tangent_pyramidnet272_8000=result["pyramidnet272"]["Tangent Attack"][8000],
        Tangent_pyramidnet272_10000=result["pyramidnet272"]["Tangent Attack"][10000],

        Tangent_gdas_1000=result["gdas"]["Tangent Attack"][1000],
        Tangent_gdas_2000=result["gdas"]["Tangent Attack"][2000],
        Tangent_gdas_5000=result["gdas"]["Tangent Attack"][5000],
        Tangent_gdas_8000=result["gdas"]["Tangent Attack"][8000],
        Tangent_gdas_10000=result["gdas"]["Tangent Attack"][10000],

        Tangent_WRN28_1000=result["WRN-28-10-drop"]["Tangent Attack"][1000],
        Tangent_WRN28_2000=result["WRN-28-10-drop"]["Tangent Attack"][2000],
        Tangent_WRN28_5000=result["WRN-28-10-drop"]["Tangent Attack"][5000],
        Tangent_WRN28_8000=result["WRN-28-10-drop"]["Tangent Attack"][8000],
        Tangent_WRN28_10000=result["WRN-28-10-drop"]["Tangent Attack"][10000],

        Tangent_WRN40_1000=result["WRN-40-10-drop"]["Tangent Attack"][1000],
        Tangent_WRN40_2000=result["WRN-40-10-drop"]["Tangent Attack"][2000],
        Tangent_WRN40_5000=result["WRN-40-10-drop"]["Tangent Attack"][5000],
        Tangent_WRN40_8000=result["WRN-40-10-drop"]["Tangent Attack"][8000],
        Tangent_WRN40_10000=result["WRN-40-10-drop"]["Tangent Attack"][10000],

        Tangent_densenet190_1000=result["densenet-bc-L190-k40"]["Tangent Attack"][1000],
        Tangent_densenet190_2000=result["densenet-bc-L190-k40"]["Tangent Attack"][2000],
        Tangent_densenet190_5000=result["densenet-bc-L190-k40"]["Tangent Attack"][5000],
        Tangent_densenet190_8000=result["densenet-bc-L190-k40"]["Tangent Attack"][8000],
        Tangent_densenet190_10000=result["densenet-bc-L190-k40"]["Tangent Attack"][10000],
    )
    )


def draw_tables_for_targeted_CIFAR_without_BBA(archs_result):
    result = archs_result
    print("""
                & & Sign-OPT & {SignOPT_pyramidnet272_1000} & {SignOPT_pyramidnet272_2000} & {SignOPT_pyramidnet272_5000} & {SignOPT_pyramidnet272_8000} & {SignOPT_pyramidnet272_10000} \\\\
                & & SVM-OPT & {SVMOPT_pyramidnet272_1000} & {SVMOPT_pyramidnet272_2000} & {SVMOPT_pyramidnet272_5000} & {SVMOPT_pyramidnet272_8000} & {SVMOPT_pyramidnet272_10000} \\\\
                & & HopSkipJumpAttack & {HSJA_pyramidnet272_1000} & {HSJA_pyramidnet272_2000} & {HSJA_pyramidnet272_5000} & {HSJA_pyramidnet272_8000} & {HSJA_pyramidnet272_10000} \\\\
                & & Tangent Attack (ours) & {Tangent_pyramidnet272_1000} & {Tangent_pyramidnet272_2000} & {Tangent_pyramidnet272_5000} & {Tangent_pyramidnet272_8000} & {Tangent_pyramidnet272_10000} \\\\
                \\cmidrule(rl){{2-12}}
                & & Sign-OPT & {SignOPT_gdas_1000} & {SignOPT_gdas_2000} & {SignOPT_gdas_5000} & {SignOPT_gdas_8000} & {SignOPT_gdas_10000} \\\\
                & & SVM-OPT & {SVMOPT_gdas_1000} & {SVMOPT_gdas_2000} & {SVMOPT_gdas_5000} & {SVMOPT_gdas_8000} & {SVMOPT_gdas_10000} \\\\
                & & HopSkipJumpAttack & {HSJA_gdas_1000} & {HSJA_gdas_2000} & {HSJA_gdas_5000} & {HSJA_gdas_8000} & {HSJA_gdas_10000} \\\\
                & & Tangent Attack (ours) & {Tangent_gdas_1000} & {Tangent_gdas_2000} & {Tangent_gdas_5000} & {Tangent_gdas_8000} & {Tangent_gdas_10000} \\\\
                \\cmidrule(rl){{2-12}}
                & & Sign-OPT & {SignOPT_WRN28_1000} & {SignOPT_WRN28_2000} & {SignOPT_WRN28_5000} & {SignOPT_WRN28_8000} & {SignOPT_WRN28_10000} \\\\
                & & SVM-OPT & {SVMOPT_WRN28_1000} & {SVMOPT_WRN28_2000} & {SVMOPT_WRN28_5000} & {SVMOPT_WRN28_8000} & {SVMOPT_WRN28_10000} \\\\
                & & HopSkipJumpAttack & {HSJA_WRN28_1000} & {HSJA_WRN28_2000} & {HSJA_WRN28_5000} & {HSJA_WRN28_8000} & {HSJA_WRN28_10000} \\\\
                & & Tangent Attack (ours) & {Tangent_WRN28_1000} & {Tangent_WRN28_2000} & {Tangent_WRN28_5000} & {Tangent_WRN28_8000} & {Tangent_WRN28_10000} \\\\
                \\cmidrule(rl){{2-12}}
                & & Sign-OPT & {SignOPT_WRN40_1000} & {SignOPT_WRN40_2000} & {SignOPT_WRN40_5000} & {SignOPT_WRN40_8000} & {SignOPT_WRN40_10000} \\\\
                & & SVM-OPT & {SVMOPT_WRN40_1000} & {SVMOPT_WRN40_2000} & {SVMOPT_WRN40_5000} & {SVMOPT_WRN40_8000} & {SVMOPT_WRN40_10000} \\\\
                & & HopSkipJumpAttack & {HSJA_WRN40_1000} & {HSJA_WRN40_2000} & {HSJA_WRN40_5000} & {HSJA_WRN40_8000} & {HSJA_WRN40_10000} \\\\
                & & Tangent Attack (ours) & {Tangent_WRN40_1000} & {Tangent_WRN40_2000} & {Tangent_WRN40_5000} & {Tangent_WRN40_8000} & {Tangent_WRN40_10000} \\\\
                \\cmidrule(rl){{2-12}}
                & & Sign-OPT & {SignOPT_densenet190_1000} & {SignOPT_densenet190_2000} & {SignOPT_densenet190_5000} & {SignOPT_densenet190_8000} & {SignOPT_densenet190_10000} \\\\
                & & SVM-OPT & {SVMOPT_densenet190_1000} & {SVMOPT_densenet190_2000} & {SVMOPT_densenet190_5000} & {SVMOPT_densenet190_8000} & {SVMOPT_densenet190_10000} \\\\
                & & HopSkipJumpAttack & {HSJA_densenet190_1000} & {HSJA_densenet190_2000} & {HSJA_densenet190_5000} & {HSJA_densenet190_8000} & {HSJA_densenet190_10000} \\\\
                & & Tangent Attack (ours) & {Tangent_densenet190_1000} & {Tangent_densenet190_2000} & {Tangent_densenet190_5000} & {Tangent_densenet190_8000} & {Tangent_densenet190_10000} \\\\
                    """.format(

        SignOPT_pyramidnet272_1000=result["pyramidnet272"]["Sign-OPT"][1000],
        SignOPT_pyramidnet272_2000=result["pyramidnet272"]["Sign-OPT"][2000],
        SignOPT_pyramidnet272_5000=result["pyramidnet272"]["Sign-OPT"][5000],
        SignOPT_pyramidnet272_8000=result["pyramidnet272"]["Sign-OPT"][8000],
        SignOPT_pyramidnet272_10000=result["pyramidnet272"]["Sign-OPT"][10000],

        SignOPT_gdas_1000=result["gdas"]["Sign-OPT"][1000],
        SignOPT_gdas_2000=result["gdas"]["Sign-OPT"][2000],
        SignOPT_gdas_5000=result["gdas"]["Sign-OPT"][5000],
        SignOPT_gdas_8000=result["gdas"]["Sign-OPT"][8000],
        SignOPT_gdas_10000=result["gdas"]["Sign-OPT"][10000],

        SignOPT_WRN28_1000=result["WRN-28-10-drop"]["Sign-OPT"][1000],
        SignOPT_WRN28_2000=result["WRN-28-10-drop"]["Sign-OPT"][2000],
        SignOPT_WRN28_5000=result["WRN-28-10-drop"]["Sign-OPT"][5000],
        SignOPT_WRN28_8000=result["WRN-28-10-drop"]["Sign-OPT"][8000],
        SignOPT_WRN28_10000=result["WRN-28-10-drop"]["Sign-OPT"][10000],

        SignOPT_WRN40_1000=result["WRN-40-10-drop"]["Sign-OPT"][1000],
        SignOPT_WRN40_2000=result["WRN-40-10-drop"]["Sign-OPT"][2000],
        SignOPT_WRN40_5000=result["WRN-40-10-drop"]["Sign-OPT"][5000],
        SignOPT_WRN40_8000=result["WRN-40-10-drop"]["Sign-OPT"][8000],
        SignOPT_WRN40_10000=result["WRN-40-10-drop"]["Sign-OPT"][10000],

        SignOPT_densenet190_1000=result["densenet-bc-L190-k40"]["Sign-OPT"][1000],
        SignOPT_densenet190_2000=result["densenet-bc-L190-k40"]["Sign-OPT"][2000],
        SignOPT_densenet190_5000=result["densenet-bc-L190-k40"]["Sign-OPT"][5000],
        SignOPT_densenet190_8000=result["densenet-bc-L190-k40"]["Sign-OPT"][8000],
        SignOPT_densenet190_10000=result["densenet-bc-L190-k40"]["Sign-OPT"][10000],

        SVMOPT_pyramidnet272_1000=result["pyramidnet272"]["SVM-OPT"][1000],
        SVMOPT_pyramidnet272_2000=result["pyramidnet272"]["SVM-OPT"][2000],
        SVMOPT_pyramidnet272_5000=result["pyramidnet272"]["SVM-OPT"][5000],
        SVMOPT_pyramidnet272_8000=result["pyramidnet272"]["SVM-OPT"][8000],
        SVMOPT_pyramidnet272_10000=result["pyramidnet272"]["SVM-OPT"][10000],

        SVMOPT_gdas_1000=result["gdas"]["SVM-OPT"][1000],
        SVMOPT_gdas_2000=result["gdas"]["SVM-OPT"][2000],
        SVMOPT_gdas_5000=result["gdas"]["SVM-OPT"][5000],
        SVMOPT_gdas_8000=result["gdas"]["SVM-OPT"][8000],
        SVMOPT_gdas_10000=result["gdas"]["SVM-OPT"][10000],

        SVMOPT_WRN28_1000=result["WRN-28-10-drop"]["SVM-OPT"][1000],
        SVMOPT_WRN28_2000=result["WRN-28-10-drop"]["SVM-OPT"][2000],
        SVMOPT_WRN28_5000=result["WRN-28-10-drop"]["SVM-OPT"][5000],
        SVMOPT_WRN28_8000=result["WRN-28-10-drop"]["SVM-OPT"][8000],
        SVMOPT_WRN28_10000=result["WRN-28-10-drop"]["SVM-OPT"][10000],

        SVMOPT_WRN40_1000=result["WRN-40-10-drop"]["SVM-OPT"][1000],
        SVMOPT_WRN40_2000=result["WRN-40-10-drop"]["SVM-OPT"][2000],
        SVMOPT_WRN40_5000=result["WRN-40-10-drop"]["SVM-OPT"][5000],
        SVMOPT_WRN40_8000=result["WRN-40-10-drop"]["SVM-OPT"][8000],
        SVMOPT_WRN40_10000=result["WRN-40-10-drop"]["SVM-OPT"][10000],

        SVMOPT_densenet190_1000=result["densenet-bc-L190-k40"]["SVM-OPT"][1000],
        SVMOPT_densenet190_2000=result["densenet-bc-L190-k40"]["SVM-OPT"][2000],
        SVMOPT_densenet190_5000=result["densenet-bc-L190-k40"]["SVM-OPT"][5000],
        SVMOPT_densenet190_8000=result["densenet-bc-L190-k40"]["SVM-OPT"][8000],
        SVMOPT_densenet190_10000=result["densenet-bc-L190-k40"]["SVM-OPT"][10000],

        HSJA_pyramidnet272_1000=result["pyramidnet272"]["HopSkipJumpAttack"][1000],
        HSJA_pyramidnet272_2000=result["pyramidnet272"]["HopSkipJumpAttack"][2000],
        HSJA_pyramidnet272_5000=result["pyramidnet272"]["HopSkipJumpAttack"][5000],
        HSJA_pyramidnet272_8000=result["pyramidnet272"]["HopSkipJumpAttack"][8000],
        HSJA_pyramidnet272_10000=result["pyramidnet272"]["HopSkipJumpAttack"][10000],

        HSJA_gdas_1000=result["gdas"]["HopSkipJumpAttack"][1000],
        HSJA_gdas_2000=result["gdas"]["HopSkipJumpAttack"][2000],
        HSJA_gdas_5000=result["gdas"]["HopSkipJumpAttack"][5000],
        HSJA_gdas_8000=result["gdas"]["HopSkipJumpAttack"][8000],
        HSJA_gdas_10000=result["gdas"]["HopSkipJumpAttack"][10000],

        HSJA_WRN28_1000=result["WRN-28-10-drop"]["HopSkipJumpAttack"][1000],
        HSJA_WRN28_2000=result["WRN-28-10-drop"]["HopSkipJumpAttack"][2000],
        HSJA_WRN28_5000=result["WRN-28-10-drop"]["HopSkipJumpAttack"][5000],
        HSJA_WRN28_8000=result["WRN-28-10-drop"]["HopSkipJumpAttack"][8000],
        HSJA_WRN28_10000=result["WRN-28-10-drop"]["HopSkipJumpAttack"][10000],

        HSJA_WRN40_1000=result["WRN-40-10-drop"]["HopSkipJumpAttack"][1000],
        HSJA_WRN40_2000=result["WRN-40-10-drop"]["HopSkipJumpAttack"][2000],
        HSJA_WRN40_5000=result["WRN-40-10-drop"]["HopSkipJumpAttack"][5000],
        HSJA_WRN40_8000=result["WRN-40-10-drop"]["HopSkipJumpAttack"][8000],
        HSJA_WRN40_10000=result["WRN-40-10-drop"]["HopSkipJumpAttack"][10000],

        HSJA_densenet190_1000=result["densenet-bc-L190-k40"]["HopSkipJumpAttack"][1000],
        HSJA_densenet190_2000=result["densenet-bc-L190-k40"]["HopSkipJumpAttack"][2000],
        HSJA_densenet190_5000=result["densenet-bc-L190-k40"]["HopSkipJumpAttack"][5000],
        HSJA_densenet190_8000=result["densenet-bc-L190-k40"]["HopSkipJumpAttack"][8000],
        HSJA_densenet190_10000=result["densenet-bc-L190-k40"]["HopSkipJumpAttack"][10000],

        Tangent_pyramidnet272_1000=result["pyramidnet272"]["Tangent Attack"][1000],
        Tangent_pyramidnet272_2000=result["pyramidnet272"]["Tangent Attack"][2000],
        Tangent_pyramidnet272_5000=result["pyramidnet272"]["Tangent Attack"][5000],
        Tangent_pyramidnet272_8000=result["pyramidnet272"]["Tangent Attack"][8000],
        Tangent_pyramidnet272_10000=result["pyramidnet272"]["Tangent Attack"][10000],

        Tangent_gdas_1000=result["gdas"]["Tangent Attack"][1000],
        Tangent_gdas_2000=result["gdas"]["Tangent Attack"][2000],
        Tangent_gdas_5000=result["gdas"]["Tangent Attack"][5000],
        Tangent_gdas_8000=result["gdas"]["Tangent Attack"][8000],
        Tangent_gdas_10000=result["gdas"]["Tangent Attack"][10000],

        Tangent_WRN28_1000=result["WRN-28-10-drop"]["Tangent Attack"][1000],
        Tangent_WRN28_2000=result["WRN-28-10-drop"]["Tangent Attack"][2000],
        Tangent_WRN28_5000=result["WRN-28-10-drop"]["Tangent Attack"][5000],
        Tangent_WRN28_8000=result["WRN-28-10-drop"]["Tangent Attack"][8000],
        Tangent_WRN28_10000=result["WRN-28-10-drop"]["Tangent Attack"][10000],

        Tangent_WRN40_1000=result["WRN-40-10-drop"]["Tangent Attack"][1000],
        Tangent_WRN40_2000=result["WRN-40-10-drop"]["Tangent Attack"][2000],
        Tangent_WRN40_5000=result["WRN-40-10-drop"]["Tangent Attack"][5000],
        Tangent_WRN40_8000=result["WRN-40-10-drop"]["Tangent Attack"][8000],
        Tangent_WRN40_10000=result["WRN-40-10-drop"]["Tangent Attack"][10000],

        Tangent_densenet190_1000=result["densenet-bc-L190-k40"]["Tangent Attack"][1000],
        Tangent_densenet190_2000=result["densenet-bc-L190-k40"]["Tangent Attack"][2000],
        Tangent_densenet190_5000=result["densenet-bc-L190-k40"]["Tangent Attack"][5000],
        Tangent_densenet190_8000=result["densenet-bc-L190-k40"]["Tangent Attack"][8000],
        Tangent_densenet190_10000=result["densenet-bc-L190-k40"]["Tangent Attack"][10000],
    )
    )


def draw_tables_for_ImageNet_untargeted(archs_result):
    result = archs_result
    print("""
            & \\multirow{{5}}{{*}}{{Inception-v3}} & Biased Boundary Attack & {BBA_inceptionv3_300} & {BBA_inceptionv3_500} & {BBA_inceptionv3_1000} & {BBA_inceptionv3_5000} & {BBA_inceptionv3_10000} & - & - & - & - \\\\
            & & Sign-OPT & {SignOPT_inceptionv3_300} & {SignOPT_inceptionv3_500} & {SignOPT_inceptionv3_1000} & {SignOPT_inceptionv3_5000} & {SignOPT_inceptionv3_10000} & - & - & - & - \\\\
            & & SVM-OPT & {SVMOPT_inceptionv3_300} & {SVMOPT_inceptionv3_500} & {SVMOPT_inceptionv3_1000} & {SVMOPT_inceptionv3_5000} & {SVMOPT_inceptionv3_10000} & - & - & - & - \\\\
            & & GeoDA & {GeoDA_inceptionv3_300} & {GeoDA_inceptionv3_500} & {GeoDA_inceptionv3_1000} & {GeoDA_inceptionv3_5000} & {GeoDA_inceptionv3_10000} & - & - & - & - \\\\
            & & RayS & {RayS_inceptionv3_300} & {RayS_inceptionv3_500} & {RayS_inceptionv3_1000} & {RayS_inceptionv3_5000} & {RayS_inceptionv3_10000} & - & - & - & - \\\\
            & & HopSkipJumpAttack & {HSJA_inceptionv3_300} & {HSJA_inceptionv3_500} & {HSJA_inceptionv3_1000} & {HSJA_inceptionv3_5000} & {HSJA_inceptionv3_10000} & - & - & - & - \\\\
            & & Tangent Attack (ours) & {Tangent_inceptionv3_300} & {Tangent_inceptionv3_500} & {Tangent_inceptionv3_1000} & {Tangent_inceptionv3_5000} & {Tangent_inceptionv3_10000} & - & - & - & - \\\\
            \\cmidrule(rl){{2-12}}
            & \\multirow{{5}}{{*}}{{Inception-v4}} & Biased Boundary Attack & {BBA_inceptionv4_300} & {BBA_inceptionv4_500} & {BBA_inceptionv4_1000} & {BBA_inceptionv4_5000} & {BBA_inceptionv4_10000} & - & - & - & - \\\\
            & & Sign-OPT & {SignOPT_inceptionv4_300} & {SignOPT_inceptionv4_500} & {SignOPT_inceptionv4_1000} & {SignOPT_inceptionv4_5000} & {SignOPT_inceptionv4_10000} & - & - & - & - \\\\
            & & SVM-OPT & {SVMOPT_inceptionv4_300} & {SVMOPT_inceptionv4_500} & {SVMOPT_inceptionv4_1000} & {SVMOPT_inceptionv4_5000} & {SVMOPT_inceptionv4_10000} & - & - & - & - \\\\
            & & GeoDA & {GeoDA_inceptionv4_300} & {GeoDA_inceptionv4_500} & {GeoDA_inceptionv4_1000} & {GeoDA_inceptionv4_5000} & {GeoDA_inceptionv4_10000} & - & - & - & - \\\\
            & & RayS & {RayS_inceptionv4_300} & {RayS_inceptionv4_500} & {RayS_inceptionv4_1000} & {RayS_inceptionv4_5000} & {RayS_inceptionv4_10000} & - & - & - & - \\\\
            & & HopSkipJumpAttack & {HSJA_inceptionv4_300} & {HSJA_inceptionv4_500} & {HSJA_inceptionv4_1000} & {HSJA_inceptionv4_5000} & {HSJA_inceptionv4_10000} & - & - & - & - \\\\
            & & Tangent Attack (ours) & {Tangent_inceptionv4_300} & {Tangent_inceptionv4_500} & {Tangent_inceptionv4_1000} & {Tangent_inceptionv4_5000} & {Tangent_inceptionv4_10000} & - & - & - & - \\\\
            \\cmidrule(rl){{2-12}}
            & \\multirow{{5}}{{*}}{{SENet-154}} & Biased Boundary Attack & {BBA_senet154_300} & {BBA_senet154_500} & {BBA_senet154_1000} & {BBA_senet154_5000} & {BBA_senet154_10000} & - & - & - & - \\\\
            & & Sign-OPT & {SignOPT_senet154_300} & {SignOPT_senet154_500} & {SignOPT_senet154_1000} & {SignOPT_senet154_5000} & {SignOPT_senet154_10000} & - & - & - & - \\\\
            & & SVM-OPT & {SVMOPT_senet154_300} & {SVMOPT_senet154_500} & {SVMOPT_senet154_1000} & {SVMOPT_senet154_5000} & {SVMOPT_senet154_10000} & - & - & - & - \\\\
            & & GeoDA & {GeoDA_senet154_300} & {GeoDA_senet154_500} & {GeoDA_senet154_1000} & {GeoDA_senet154_5000} & {GeoDA_senet154_10000} & - & - & - & - \\\\
            & & RayS & {RayS_senet154_300} & {RayS_senet154_500} & {RayS_senet154_1000} & {RayS_senet154_5000} & {RayS_senet154_10000} & - & - & - & - \\\\
            & & HopSkipJumpAttack & {HSJA_senet154_300} & {HSJA_senet154_500} & {HSJA_senet154_1000} & {HSJA_senet154_5000} & {HSJA_senet154_10000} & - & - & - & - \\\\
            & & Tangent Attack (ours) & {Tangent_senet154_300} & {Tangent_senet154_500} & {Tangent_senet154_1000} & {Tangent_senet154_5000} & {Tangent_senet154_10000} & - & - & - & - \\\\
            \\cmidrule(rl){{2-12}}
            & \\multirow{{5}}{{*}}{{ResNet-101}} & Biased Boundary Attack & {BBA_resnet101_300} & {BBA_resnet101_500} & {BBA_resnet101_1000} & {BBA_resnet101_5000} & {BBA_resnet101_10000} & - & - & - & - \\\\
            & & Sign-OPT & {SignOPT_resnet101_300} & {SignOPT_resnet101_500} & {SignOPT_resnet101_1000} & {SignOPT_resnet101_5000} & {SignOPT_resnet101_10000} & - & - & - & - \\\\
            & & SVM-OPT & {SVMOPT_resnet101_300} & {SVMOPT_resnet101_500} & {SVMOPT_resnet101_1000} & {SVMOPT_resnet101_5000} & {SVMOPT_resnet101_10000} & - & - & - & - \\\\
            & & GeoDA & {GeoDA_resnet101_300} & {GeoDA_resnet101_500} & {GeoDA_resnet101_1000} & {GeoDA_resnet101_5000} & {GeoDA_resnet101_10000} & - & - & - & - \\\\
            & & RayS & {RayS_resnet101_300} & {RayS_resnet101_500} & {RayS_resnet101_1000} & {RayS_resnet101_5000} & {RayS_resnet101_10000} & - & - & - & - \\\\
            & & HopSkipJumpAttack & {HSJA_resnet101_300} & {HSJA_resnet101_500} & {HSJA_resnet101_1000} & {HSJA_resnet101_5000} & {HSJA_resnet101_10000} & - & - & - & - \\\\
            & & Tangent Attack (ours) & {Tangent_resnet101_300} & {Tangent_resnet101_500} & {Tangent_resnet101_1000} & {Tangent_resnet101_5000} & {Tangent_resnet101_10000} & - & - & - & - \\\\
            \\cmidrule(rl){{2-12}}
            & \\multirow{{5}}{{*}}{{ResNeXt-101}} & Biased Boundary Attack & {BBA_resnext101_300} & {BBA_resnext101_500} & {BBA_resnext101_1000} & {BBA_resnext101_5000} & {BBA_resnext101_10000} & - & - & - & - \\\\
            & & Sign-OPT & {SignOPT_resnext101_300} & {SignOPT_resnext101_500} & {SignOPT_resnext101_1000} & {SignOPT_resnext101_5000} & {SignOPT_resnext101_10000} & - & - & - & - \\\\
            & & SVM-OPT & {SVMOPT_resnext101_300} & {SVMOPT_resnext101_500} & {SVMOPT_resnext101_1000} & {SVMOPT_resnext101_5000} & {SVMOPT_resnext101_10000} & - & - & - & - \\\\
            & & GeoDA & {GeoDA_resnext101_300} & {GeoDA_resnext101_500} & {GeoDA_resnext101_1000} & {GeoDA_resnext101_5000} & {GeoDA_resnext101_10000} & - & - & - & - \\\\
            & & RayS & {RayS_resnext101_300} & {RayS_resnext101_500} & {RayS_resnext101_1000} & {RayS_resnext101_5000} & {RayS_resnext101_10000} & - & - & - & - \\\\
            & & HopSkipJumpAttack & {HSJA_resnext101_300} & {HSJA_resnext101_500} & {HSJA_resnext101_1000} & {HSJA_resnext101_5000} & {HSJA_resnext101_10000} & - & - & - & - \\\\
            & & Tangent Attack (ours) & {Tangent_resnext101_300} & {Tangent_resnext101_500} & {Tangent_resnext101_1000} & {Tangent_resnext101_5000} & {Tangent_resnext101_10000} & - & - & - & - \\\\
                """.format(
        BBA_inceptionv3_300=result["inceptionv3"]["Biased Boundary Attack"][300],
        BBA_inceptionv3_500=result["inceptionv3"]["Biased Boundary Attack"][500],
        BBA_inceptionv3_1000=result["inceptionv3"]["Biased Boundary Attack"][1000],
        BBA_inceptionv3_5000=result["inceptionv3"]["Biased Boundary Attack"][5000],
        BBA_inceptionv3_10000=result["inceptionv3"]["Biased Boundary Attack"][10000],

        BBA_inceptionv4_300=result["inceptionv4"]["Biased Boundary Attack"][300],
        BBA_inceptionv4_500=result["inceptionv4"]["Biased Boundary Attack"][500],
        BBA_inceptionv4_1000=result["inceptionv4"]["Biased Boundary Attack"][1000],
        BBA_inceptionv4_5000=result["inceptionv4"]["Biased Boundary Attack"][5000],
        BBA_inceptionv4_10000=result["inceptionv4"]["Biased Boundary Attack"][10000],

        BBA_senet154_300=result["senet154"]["Biased Boundary Attack"][300],
        BBA_senet154_500=result["senet154"]["Biased Boundary Attack"][500],
        BBA_senet154_1000=result["senet154"]["Biased Boundary Attack"][1000],
        BBA_senet154_5000=result["senet154"]["Biased Boundary Attack"][5000],
        BBA_senet154_10000=result["senet154"]["Biased Boundary Attack"][10000],

        BBA_resnet101_300=result["resnet101"]["Biased Boundary Attack"][300],
        BBA_resnet101_500=result["resnet101"]["Biased Boundary Attack"][500],
        BBA_resnet101_1000=result["resnet101"]["Biased Boundary Attack"][1000],
        BBA_resnet101_5000=result["resnet101"]["Biased Boundary Attack"][5000],
        BBA_resnet101_10000=result["resnet101"]["Biased Boundary Attack"][10000],

        BBA_resnext101_300=result["resnext101_64x4d"]["Biased Boundary Attack"][300],
        BBA_resnext101_500=result["resnext101_64x4d"]["Biased Boundary Attack"][500],
        BBA_resnext101_1000=result["resnext101_64x4d"]["Biased Boundary Attack"][1000],
        BBA_resnext101_5000=result["resnext101_64x4d"]["Biased Boundary Attack"][5000],
        BBA_resnext101_10000=result["resnext101_64x4d"]["Biased Boundary Attack"][10000],

        SignOPT_inceptionv3_300=result["inceptionv3"]["Sign-OPT"][300],
        SignOPT_inceptionv3_500=result["inceptionv3"]["Sign-OPT"][500],
        SignOPT_inceptionv3_1000=result["inceptionv3"]["Sign-OPT"][1000],
        SignOPT_inceptionv3_5000=result["inceptionv3"]["Sign-OPT"][5000],
        SignOPT_inceptionv3_10000=result["inceptionv3"]["Sign-OPT"][10000],

        SignOPT_inceptionv4_300=result["inceptionv4"]["Sign-OPT"][300],
        SignOPT_inceptionv4_500=result["inceptionv4"]["Sign-OPT"][500],
        SignOPT_inceptionv4_1000=result["inceptionv4"]["Sign-OPT"][1000],
        SignOPT_inceptionv4_5000=result["inceptionv4"]["Sign-OPT"][5000],
        SignOPT_inceptionv4_10000=result["inceptionv4"]["Sign-OPT"][10000],

        SignOPT_senet154_300=result["senet154"]["Sign-OPT"][300],
        SignOPT_senet154_500=result["senet154"]["Sign-OPT"][500],
        SignOPT_senet154_1000=result["senet154"]["Sign-OPT"][1000],
        SignOPT_senet154_5000=result["senet154"]["Sign-OPT"][5000],
        SignOPT_senet154_10000=result["senet154"]["Sign-OPT"][10000],

        SignOPT_resnet101_300=result["resnet101"]["Sign-OPT"][300],
        SignOPT_resnet101_500=result["resnet101"]["Sign-OPT"][500],
        SignOPT_resnet101_1000=result["resnet101"]["Sign-OPT"][1000],
        SignOPT_resnet101_5000=result["resnet101"]["Sign-OPT"][5000],
        SignOPT_resnet101_10000=result["resnet101"]["Sign-OPT"][10000],

        SignOPT_resnext101_300=result["resnext101_64x4d"]["Sign-OPT"][300],
        SignOPT_resnext101_500=result["resnext101_64x4d"]["Sign-OPT"][500],
        SignOPT_resnext101_1000=result["resnext101_64x4d"]["Sign-OPT"][1000],
        SignOPT_resnext101_5000=result["resnext101_64x4d"]["Sign-OPT"][5000],
        SignOPT_resnext101_10000=result["resnext101_64x4d"]["Sign-OPT"][10000],

        SVMOPT_inceptionv3_300=result["inceptionv3"]["SVM-OPT"][300],
        SVMOPT_inceptionv3_500=result["inceptionv3"]["SVM-OPT"][500],
        SVMOPT_inceptionv3_1000=result["inceptionv3"]["SVM-OPT"][1000],
        SVMOPT_inceptionv3_5000=result["inceptionv3"]["SVM-OPT"][5000],
        SVMOPT_inceptionv3_10000=result["inceptionv3"]["SVM-OPT"][10000],

        SVMOPT_inceptionv4_300=result["inceptionv4"]["SVM-OPT"][300],
        SVMOPT_inceptionv4_500=result["inceptionv4"]["SVM-OPT"][500],
        SVMOPT_inceptionv4_1000=result["inceptionv4"]["SVM-OPT"][1000],
        SVMOPT_inceptionv4_5000=result["inceptionv4"]["SVM-OPT"][5000],
        SVMOPT_inceptionv4_10000=result["inceptionv4"]["SVM-OPT"][10000],

        SVMOPT_senet154_300=result["senet154"]["SVM-OPT"][300],
        SVMOPT_senet154_500=result["senet154"]["SVM-OPT"][500],
        SVMOPT_senet154_1000=result["senet154"]["SVM-OPT"][1000],
        SVMOPT_senet154_5000=result["senet154"]["SVM-OPT"][5000],
        SVMOPT_senet154_10000=result["senet154"]["SVM-OPT"][10000],

        SVMOPT_resnet101_300=result["resnet101"]["SVM-OPT"][300],
        SVMOPT_resnet101_500=result["resnet101"]["SVM-OPT"][500],
        SVMOPT_resnet101_1000=result["resnet101"]["SVM-OPT"][1000],
        SVMOPT_resnet101_5000=result["resnet101"]["SVM-OPT"][5000],
        SVMOPT_resnet101_10000=result["resnet101"]["SVM-OPT"][10000],

        SVMOPT_resnext101_300=result["resnext101_64x4d"]["SVM-OPT"][300],
        SVMOPT_resnext101_500=result["resnext101_64x4d"]["SVM-OPT"][500],
        SVMOPT_resnext101_1000=result["resnext101_64x4d"]["SVM-OPT"][1000],
        SVMOPT_resnext101_5000=result["resnext101_64x4d"]["SVM-OPT"][5000],
        SVMOPT_resnext101_10000=result["resnext101_64x4d"]["SVM-OPT"][10000],

        GeoDA_inceptionv3_300=result["inceptionv3"]["GeoDA"][300],
        GeoDA_inceptionv3_500=result["inceptionv3"]["GeoDA"][500],
        GeoDA_inceptionv3_1000=result["inceptionv3"]["GeoDA"][1000],
        GeoDA_inceptionv3_5000=result["inceptionv3"]["GeoDA"][5000],
        GeoDA_inceptionv3_10000=result["inceptionv3"]["GeoDA"][10000],

        GeoDA_inceptionv4_300=result["inceptionv4"]["GeoDA"][300],
        GeoDA_inceptionv4_500=result["inceptionv4"]["GeoDA"][500],
        GeoDA_inceptionv4_1000=result["inceptionv4"]["GeoDA"][1000],
        GeoDA_inceptionv4_5000=result["inceptionv4"]["GeoDA"][5000],
        GeoDA_inceptionv4_10000=result["inceptionv4"]["GeoDA"][10000],

        GeoDA_senet154_300=result["senet154"]["GeoDA"][300],
        GeoDA_senet154_500=result["senet154"]["GeoDA"][500],
        GeoDA_senet154_1000=result["senet154"]["GeoDA"][1000],
        GeoDA_senet154_5000=result["senet154"]["GeoDA"][5000],
        GeoDA_senet154_10000=result["senet154"]["GeoDA"][10000],

        GeoDA_resnet101_300=result["resnet101"]["GeoDA"][300],
        GeoDA_resnet101_500=result["resnet101"]["GeoDA"][500],
        GeoDA_resnet101_1000=result["resnet101"]["GeoDA"][1000],
        GeoDA_resnet101_5000=result["resnet101"]["GeoDA"][5000],
        GeoDA_resnet101_10000=result["resnet101"]["GeoDA"][10000],

        GeoDA_resnext101_300=result["resnext101_64x4d"]["GeoDA"][300],
        GeoDA_resnext101_500=result["resnext101_64x4d"]["GeoDA"][500],
        GeoDA_resnext101_1000=result["resnext101_64x4d"]["GeoDA"][1000],
        GeoDA_resnext101_5000=result["resnext101_64x4d"]["GeoDA"][5000],
        GeoDA_resnext101_10000=result["resnext101_64x4d"]["GeoDA"][10000],

        RayS_inceptionv3_300=result["inceptionv3"]["RayS"][300],
        RayS_inceptionv3_500=result["inceptionv3"]["RayS"][500],
        RayS_inceptionv3_1000=result["inceptionv3"]["RayS"][1000],
        RayS_inceptionv3_5000=result["inceptionv3"]["RayS"][5000],
        RayS_inceptionv3_10000=result["inceptionv3"]["RayS"][10000],

        RayS_inceptionv4_300=result["inceptionv4"]["RayS"][300],
        RayS_inceptionv4_500=result["inceptionv4"]["RayS"][500],
        RayS_inceptionv4_1000=result["inceptionv4"]["RayS"][1000],
        RayS_inceptionv4_5000=result["inceptionv4"]["RayS"][5000],
        RayS_inceptionv4_10000=result["inceptionv4"]["RayS"][10000],

        RayS_senet154_300=result["senet154"]["RayS"][300],
        RayS_senet154_500=result["senet154"]["RayS"][500],
        RayS_senet154_1000=result["senet154"]["RayS"][1000],
        RayS_senet154_5000=result["senet154"]["RayS"][5000],
        RayS_senet154_10000=result["senet154"]["RayS"][10000],

        RayS_resnet101_300=result["resnet101"]["RayS"][300],
        RayS_resnet101_500=result["resnet101"]["RayS"][500],
        RayS_resnet101_1000=result["resnet101"]["RayS"][1000],
        RayS_resnet101_5000=result["resnet101"]["RayS"][5000],
        RayS_resnet101_10000=result["resnet101"]["RayS"][10000],

        RayS_resnext101_300=result["resnext101_64x4d"]["RayS"][300],
        RayS_resnext101_500=result["resnext101_64x4d"]["RayS"][500],
        RayS_resnext101_1000=result["resnext101_64x4d"]["RayS"][1000],
        RayS_resnext101_5000=result["resnext101_64x4d"]["RayS"][5000],
        RayS_resnext101_10000=result["resnext101_64x4d"]["RayS"][10000],

        HSJA_inceptionv3_300=result["inceptionv3"]["HopSkipJumpAttack"][300],
        HSJA_inceptionv3_500=result["inceptionv3"]["HopSkipJumpAttack"][500],
        HSJA_inceptionv3_1000=result["inceptionv3"]["HopSkipJumpAttack"][1000],
        HSJA_inceptionv3_5000=result["inceptionv3"]["HopSkipJumpAttack"][5000],
        HSJA_inceptionv3_10000=result["inceptionv3"]["HopSkipJumpAttack"][10000],

        HSJA_inceptionv4_300=result["inceptionv4"]["HopSkipJumpAttack"][300],
        HSJA_inceptionv4_500=result["inceptionv4"]["HopSkipJumpAttack"][500],
        HSJA_inceptionv4_1000=result["inceptionv4"]["HopSkipJumpAttack"][1000],
        HSJA_inceptionv4_5000=result["inceptionv4"]["HopSkipJumpAttack"][5000],
        HSJA_inceptionv4_10000=result["inceptionv4"]["HopSkipJumpAttack"][10000],

        HSJA_senet154_300=result["senet154"]["HopSkipJumpAttack"][300],
        HSJA_senet154_500=result["senet154"]["HopSkipJumpAttack"][500],
        HSJA_senet154_1000=result["senet154"]["HopSkipJumpAttack"][1000],
        HSJA_senet154_5000=result["senet154"]["HopSkipJumpAttack"][5000],
        HSJA_senet154_10000=result["senet154"]["HopSkipJumpAttack"][10000],

        HSJA_resnet101_300=result["resnet101"]["HopSkipJumpAttack"][300],
        HSJA_resnet101_500=result["resnet101"]["HopSkipJumpAttack"][500],
        HSJA_resnet101_1000=result["resnet101"]["HopSkipJumpAttack"][1000],
        HSJA_resnet101_5000=result["resnet101"]["HopSkipJumpAttack"][5000],
        HSJA_resnet101_10000=result["resnet101"]["HopSkipJumpAttack"][10000],

        HSJA_resnext101_300=result["resnext101_64x4d"]["HopSkipJumpAttack"][300],
        HSJA_resnext101_500=result["resnext101_64x4d"]["HopSkipJumpAttack"][500],
        HSJA_resnext101_1000=result["resnext101_64x4d"]["HopSkipJumpAttack"][1000],
        HSJA_resnext101_5000=result["resnext101_64x4d"]["HopSkipJumpAttack"][5000],
        HSJA_resnext101_10000=result["resnext101_64x4d"]["HopSkipJumpAttack"][10000],

        Tangent_inceptionv3_300=result["inceptionv3"]["Tangent Attack"][300],
        Tangent_inceptionv3_500=result["inceptionv3"]["Tangent Attack"][500],
        Tangent_inceptionv3_1000=result["inceptionv3"]["Tangent Attack"][1000],
        Tangent_inceptionv3_5000=result["inceptionv3"]["Tangent Attack"][5000],
        Tangent_inceptionv3_10000=result["inceptionv3"]["Tangent Attack"][10000],

        Tangent_inceptionv4_300=result["inceptionv4"]["Tangent Attack"][300],
        Tangent_inceptionv4_500=result["inceptionv4"]["Tangent Attack"][500],
        Tangent_inceptionv4_1000=result["inceptionv4"]["Tangent Attack"][1000],
        Tangent_inceptionv4_5000=result["inceptionv4"]["Tangent Attack"][5000],
        Tangent_inceptionv4_10000=result["inceptionv4"]["Tangent Attack"][10000],

        Tangent_senet154_300=result["senet154"]["Tangent Attack"][300],
        Tangent_senet154_500=result["senet154"]["Tangent Attack"][500],
        Tangent_senet154_1000=result["senet154"]["Tangent Attack"][1000],
        Tangent_senet154_5000=result["senet154"]["Tangent Attack"][5000],
        Tangent_senet154_10000=result["senet154"]["Tangent Attack"][10000],

        Tangent_resnet101_300=result["resnet101"]["Tangent Attack"][300],
        Tangent_resnet101_500=result["resnet101"]["Tangent Attack"][500],
        Tangent_resnet101_1000=result["resnet101"]["Tangent Attack"][1000],
        Tangent_resnet101_5000=result["resnet101"]["Tangent Attack"][5000],
        Tangent_resnet101_10000=result["resnet101"]["Tangent Attack"][10000],

        Tangent_resnext101_300=result["resnext101_64x4d"]["Tangent Attack"][300],
        Tangent_resnext101_500=result["resnext101_64x4d"]["Tangent Attack"][500],
        Tangent_resnext101_1000=result["resnext101_64x4d"]["Tangent Attack"][1000],
        Tangent_resnext101_5000=result["resnext101_64x4d"]["Tangent Attack"][5000],
        Tangent_resnext101_10000=result["resnext101_64x4d"]["Tangent Attack"][10000],
    )
    )
if __name__ == "__main__":
    dataset = "CIFAR-10"
    norm = "l2"
    if "CIFAR" in dataset:
        archs = ['pyramidnet272',"gdas","WRN-28-10-drop", "WRN-40-10-drop"]
    else:
        archs = ["inceptionv4","senet154","resnet101","inceptionv3"]
    query_budgets = [300, 1000,2000,5000,8000,10000]
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

        draw_wide_table_CIFAR_with_boundary(untargeted_result, targeted_result)
    else:
        result_archs = {}
        targeted_result_archs = {}
        for arch in archs:
            result = fetch_all_json_content_given_contraint(dataset, norm, False, arch, query_budgets, "mean_distortion")
            result_archs[arch] = result
            result = fetch_all_json_content_given_contraint(dataset, norm, True, arch, query_budgets,
                                                            "mean_distortion")
            targeted_result_archs[arch] = result

        draw_targeted_first_tables_for_ImageNet_with_300_query(result_archs, targeted_result_archs)