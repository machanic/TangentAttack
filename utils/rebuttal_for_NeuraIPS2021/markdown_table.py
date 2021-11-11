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


method_name_to_paper = {"tangent_attack":"Tangent Attack(hemisphere)",  "HSJA":"HSJA",
                        "ellipsoid_tangent_attack":"Tangent Attack(semiellipsoid)",
                        "SignOPT":"Sign-OPT", "SVMOPT":"SVM-OPT",
                        "boundary_attack":"Boundary Attack",
                        "GeoDA":"GeoDA",
                        "RayS":"RayS",
                        "QEBA":"QEBA",
                        "QEBATangentAttack":"QEBATangentAttack"
                        }
                        # "RayS": "RayS","GeoDA": "GeoDA",
                        #"biased_boundary_attack": "Biased Boundary Attack"}

def from_method_to_dir_path(dataset, method, norm, targeted,target_type):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if method == "tangent_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str=target_str)
    elif method == "ellipsoid_tangent_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                               norm=norm,
                                                               target_str=target_str)
    elif method == "HSJA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm,  target_str=target_str)
    if method == "tangent_attack@30":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str=target_str)
    elif method == "HSJA@30":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm,  target_str=target_str)
    elif method == "GeoDA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str=target_str)
    elif method == "QEBA":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                               norm=norm, target_str=target_str)
    elif method == "QEBATangentAttack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                               norm=norm, target_str=target_str)
    elif method == "biased_boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str=target_str)
    elif method == "boundary_attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str=target_str)
    elif method == "RayS":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str=target_str)
    elif method == "SignOPT":
        if targeted:
            path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                                   target_str=target_str)
        else:
            path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str=target_str)
    elif method == "SVMOPT":
        if targeted:
            path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                                   target_str=target_str)
        else:
            path = "{method}-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str=target_str)
    return path


def read_json_and_extract(json_path):
    with open(json_path, "r") as file_obj:
        json_content = json.load(file_obj)
        distortion = json_content["distortion"]
        return distortion

def get_file_name_list(dataset, method_name_to_paper, norm, targeted, target_type):
    folder_path_dict = {}
    for method, paper_method_name in method_name_to_paper.items():
        file_path = "/home1/machen/hard_label_attacks/logs/" + from_method_to_dir_path(dataset, method, norm, targeted,target_type)
        folder_path_dict[paper_method_name] = file_path
    return folder_path_dict



def get_mean_and_median_distortion_given_query_budgets(distortion_dict, query_budgets, want_key):
    mean_and_median_distortions = defaultdict(lambda : "-")
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


def fetch_all_json_content_given_contraint(dataset, norm, targeted, target_type, arch, query_budgets, want_key="mean_distortion"):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm, targeted,target_type)
    result = {}
    for method, folder in folder_list.items():
        if "QEBA" in method:
            file_path = folder + "/{}_pgen_resize_result.json".format(arch)
        else:
            file_path = folder + "/{}_result.json".format(arch)
        if method in ["RayS","GeoDA"] and targeted:
            print("{} does not exist!".format(file_path))
            result[method] = defaultdict(lambda : "-")
            continue
        if not os.path.exists(file_path):
            distortion_dict = {}
        else:
            distortion_dict = read_json_and_extract(file_path)
        print(file_path)
        mean_and_median_distortions = get_mean_and_median_distortion_given_query_budgets(distortion_dict, query_budgets,want_key)
        result[method] = mean_and_median_distortions
    return result


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

def draw_tables_for_ImageNet_10K_to_20K(result):
    print("""
            | Method | @10K | @13K | @15K | @17K | @19K | @20K |
            | :-| :- | :- |:- |:- |:- | :- |
            | Boundary Attack | {BA_resnet101_300} | {BA_resnet101_1000} | {BA_resnet101_2000} | {BA_resnet101_5000} | {BA_resnet101_8000} |{BA_resnet101_10000} |
            | Sign-OPT | {SignOPT_resnet101_300} | {SignOPT_resnet101_1000} | {SignOPT_resnet101_2000} | {SignOPT_resnet101_5000} | {SignOPT_resnet101_8000} | {SignOPT_resnet101_10000} |
            | SVM-OPT | {SVMOPT_resnet101_300} | {SVMOPT_resnet101_1000} | {SVMOPT_resnet101_2000} | {SVMOPT_resnet101_5000} | {SVMOPT_resnet101_8000} | {SVMOPT_resnet101_10000} |
            | GeoDA | {GeoDA_resnet101_300} | {GeoDA_resnet101_1000} | {GeoDA_resnet101_2000} | {GeoDA_resnet101_5000} | {GeoDA_resnet101_8000} | {GeoDA_resnet101_10000} |
            | RayS | {RayS_resnet101_300} | {RayS_resnet101_1000} | {RayS_resnet101_2000} | {RayS_resnet101_5000} | {RayS_resnet101_8000} | {RayS_resnet101_10000} |
            | HSJA | {HSJA_resnet101_300} | {HSJA_resnet101_1000} | {HSJA_resnet101_2000} | {HSJA_resnet101_5000} | {HSJA_resnet101_8000} |  {HSJA_resnet101_10000} |
            | QEBA-S | {QEBA_resnet101_300} | {QEBA_resnet101_1000} | {QEBA_resnet101_2000} | {QEBA_resnet101_5000} | {QEBA_resnet101_8000} | {QEBA_resnet101_10000} |
            | Tangent Attack(hemisphere) | {TangentAttackHemisphere_resnet101_300} | {TangentAttackHemisphere_resnet101_1000} | {TangentAttackHemisphere_resnet101_2000} | {TangentAttackHemisphere_resnet101_5000} | {TangentAttackHemisphere_resnet101_8000} | {TangentAttackHemisphere_resnet101_10000} |
            | Tangent Attack(semiellipsoid) | {TangentAttackSemiellipsoid_resnet101_300} | {TangentAttackSemiellipsoid_resnet101_1000} | {TangentAttackSemiellipsoid_resnet101_2000} | {TangentAttackSemiellipsoid_resnet101_5000} | {TangentAttackSemiellipsoid_resnet101_8000} | {TangentAttackSemiellipsoid_resnet101_10000} |
            | QEBA+Tangent Attack | {QEBATangentAttack_resnet101_300} | {QEBATangentAttack_resnet101_1000} | {QEBATangentAttack_resnet101_2000} | {QEBATangentAttack_resnet101_5000} | {QEBATangentAttack_resnet101_8000} | {QEBATangentAttack_resnet101_10000} |
                            """.format(
        BA_resnet101_300=result["resnet101"]["Boundary Attack"][10000],
        BA_resnet101_1000=result["resnet101"]["Boundary Attack"][13000],
        BA_resnet101_2000=result["resnet101"]["Boundary Attack"][15000],
        BA_resnet101_5000=result["resnet101"]["Boundary Attack"][17000],
        BA_resnet101_8000=result["resnet101"]["Boundary Attack"][19000],
        BA_resnet101_10000=result["resnet101"]["Boundary Attack"][20000],

        SignOPT_resnet101_300=result["resnet101"]["Sign-OPT"][10000],
        SignOPT_resnet101_1000=result["resnet101"]["Sign-OPT"][13000],
        SignOPT_resnet101_2000=result["resnet101"]["Sign-OPT"][15000],
        SignOPT_resnet101_5000=result["resnet101"]["Sign-OPT"][17000],
        SignOPT_resnet101_8000=result["resnet101"]["Sign-OPT"][19000],
        SignOPT_resnet101_10000=result["resnet101"]["Sign-OPT"][20000],

        SVMOPT_resnet101_300=result["resnet101"]["SVM-OPT"][10000],
        SVMOPT_resnet101_1000=result["resnet101"]["SVM-OPT"][13000],
        SVMOPT_resnet101_2000=result["resnet101"]["SVM-OPT"][15000],
        SVMOPT_resnet101_5000=result["resnet101"]["SVM-OPT"][17000],
        SVMOPT_resnet101_8000=result["resnet101"]["SVM-OPT"][19000],
        SVMOPT_resnet101_10000=result["resnet101"]["SVM-OPT"][20000],

        GeoDA_resnet101_300=result["resnet101"]["GeoDA"][10000],
        GeoDA_resnet101_1000=result["resnet101"]["GeoDA"][13000],
        GeoDA_resnet101_2000=result["resnet101"]["GeoDA"][15000],
        GeoDA_resnet101_5000=result["resnet101"]["GeoDA"][17000],
        GeoDA_resnet101_8000=result["resnet101"]["GeoDA"][19000],
        GeoDA_resnet101_10000=result["resnet101"]["GeoDA"][20000],

        RayS_resnet101_300=result["resnet101"]["RayS"][10000],
        RayS_resnet101_1000=result["resnet101"]["RayS"][13000],
        RayS_resnet101_2000=result["resnet101"]["RayS"][15000],
        RayS_resnet101_5000=result["resnet101"]["RayS"][17000],
        RayS_resnet101_8000=result["resnet101"]["RayS"][19000],
        RayS_resnet101_10000=result["resnet101"]["RayS"][20000],

        HSJA_resnet101_300=result["resnet101"]["HSJA"][10000],
        HSJA_resnet101_1000=result["resnet101"]["HSJA"][13000],
        HSJA_resnet101_2000=result["resnet101"]["HSJA"][15000],
        HSJA_resnet101_5000=result["resnet101"]["HSJA"][17000],
        HSJA_resnet101_8000=result["resnet101"]["HSJA"][19000],
        HSJA_resnet101_10000=result["resnet101"]["HSJA"][20000],

        QEBA_resnet101_300=result["resnet101"]["QEBA"][10000],
        QEBA_resnet101_1000=result["resnet101"]["QEBA"][13000],
        QEBA_resnet101_2000=result["resnet101"]["QEBA"][15000],
        QEBA_resnet101_5000=result["resnet101"]["QEBA"][17000],
        QEBA_resnet101_8000=result["resnet101"]["QEBA"][19000],
        QEBA_resnet101_10000=result["resnet101"]["QEBA"][20000],

        TangentAttackHemisphere_resnet101_300=result["resnet101"]["Tangent Attack(hemisphere)"][10000],
        TangentAttackHemisphere_resnet101_1000=result["resnet101"]["Tangent Attack(hemisphere)"][13000],
        TangentAttackHemisphere_resnet101_2000=result["resnet101"]["Tangent Attack(hemisphere)"][15000],
        TangentAttackHemisphere_resnet101_5000=result["resnet101"]["Tangent Attack(hemisphere)"][17000],
        TangentAttackHemisphere_resnet101_8000=result["resnet101"]["Tangent Attack(hemisphere)"][19000],
        TangentAttackHemisphere_resnet101_10000=result["resnet101"]["Tangent Attack(hemisphere)"][20000],

        TangentAttackSemiellipsoid_resnet101_300=result["resnet101"]["Tangent Attack(semiellipsoid)"][10000],
        TangentAttackSemiellipsoid_resnet101_1000=result["resnet101"]["Tangent Attack(semiellipsoid)"][13000],
        TangentAttackSemiellipsoid_resnet101_2000=result["resnet101"]["Tangent Attack(semiellipsoid)"][15000],
        TangentAttackSemiellipsoid_resnet101_5000=result["resnet101"]["Tangent Attack(semiellipsoid)"][17000],
        TangentAttackSemiellipsoid_resnet101_8000=result["resnet101"]["Tangent Attack(semiellipsoid)"][19000],
        TangentAttackSemiellipsoid_resnet101_10000=result["resnet101"]["Tangent Attack(semiellipsoid)"][20000],

        QEBATangentAttack_resnet101_300=result["resnet101"]["QEBATangentAttack"][10000],
        QEBATangentAttack_resnet101_1000=result["resnet101"]["QEBATangentAttack"][13000],
        QEBATangentAttack_resnet101_2000=result["resnet101"]["QEBATangentAttack"][15000],
        QEBATangentAttack_resnet101_5000=result["resnet101"]["QEBATangentAttack"][17000],
        QEBATangentAttack_resnet101_8000=result["resnet101"]["QEBATangentAttack"][19000],
        QEBATangentAttack_resnet101_10000=result["resnet101"]["QEBATangentAttack"][20000],
    )
    )


def draw_tables_for_ImageNet(result):
    print("""
            | Method | @300 | @1K | @2K | @5K | @8K | @10K |
            | :-| :- | :- |:- |:- |:- | :- |
            | Boundary Attack | {BA_resnet101_300} | {BA_resnet101_1000} | {BA_resnet101_2000} | {BA_resnet101_5000} | {BA_resnet101_8000} |{BA_resnet101_10000} |
            | Sign-OPT | {SignOPT_resnet101_300} | {SignOPT_resnet101_1000} | {SignOPT_resnet101_2000} | {SignOPT_resnet101_5000} | {SignOPT_resnet101_8000} | {SignOPT_resnet101_10000} |
            | SVM-OPT | {SVMOPT_resnet101_300} | {SVMOPT_resnet101_1000} | {SVMOPT_resnet101_2000} | {SVMOPT_resnet101_5000} | {SVMOPT_resnet101_8000} | {SVMOPT_resnet101_10000} |
            | GeoDA | {GeoDA_resnet101_300} | {GeoDA_resnet101_1000} | {GeoDA_resnet101_2000} | {GeoDA_resnet101_5000} | {GeoDA_resnet101_8000} | {GeoDA_resnet101_10000} |
            | RayS | {RayS_resnet101_300} | {RayS_resnet101_1000} | {RayS_resnet101_2000} | {RayS_resnet101_5000} | {RayS_resnet101_8000} | {RayS_resnet101_10000} |
            | HSJA | {HSJA_resnet101_300} | {HSJA_resnet101_1000} | {HSJA_resnet101_2000} | {HSJA_resnet101_5000} | {HSJA_resnet101_8000} |  {HSJA_resnet101_10000} |
            | QEBA-S | {QEBA_resnet101_300} | {QEBA_resnet101_1000} | {QEBA_resnet101_2000} | {QEBA_resnet101_5000} | {QEBA_resnet101_8000} | {QEBA_resnet101_10000} |
            | Tangent Attack(hemisphere) | {TangentAttackHemisphere_resnet101_300} | {TangentAttackHemisphere_resnet101_1000} | {TangentAttackHemisphere_resnet101_2000} | {TangentAttackHemisphere_resnet101_5000} | {TangentAttackHemisphere_resnet101_8000} | {TangentAttackHemisphere_resnet101_10000} |
            | Tangent Attack(semiellipsoid) | {TangentAttackSemiellipsoid_resnet101_300} | {TangentAttackSemiellipsoid_resnet101_1000} | {TangentAttackSemiellipsoid_resnet101_2000} | {TangentAttackSemiellipsoid_resnet101_5000} | {TangentAttackSemiellipsoid_resnet101_8000} | {TangentAttackSemiellipsoid_resnet101_10000} |
            | QEBA+Tangent Attack | {QEBATangentAttack_resnet101_300} | {QEBATangentAttack_resnet101_1000} | {QEBATangentAttack_resnet101_2000} | {QEBATangentAttack_resnet101_5000} | {QEBATangentAttack_resnet101_8000} | {QEBATangentAttack_resnet101_10000} |
                            """.format(
        BA_resnet101_300=result["resnet101"]["Boundary Attack"][300],
        BA_resnet101_1000=result["resnet101"]["Boundary Attack"][1000],
        BA_resnet101_2000=result["resnet101"]["Boundary Attack"][2000],
        BA_resnet101_5000=result["resnet101"]["Boundary Attack"][5000],
        BA_resnet101_8000=result["resnet101"]["Boundary Attack"][8000],
        BA_resnet101_10000=result["resnet101"]["Boundary Attack"][10000],

        SignOPT_resnet101_300=result["resnet101"]["Sign-OPT"][300],
        SignOPT_resnet101_1000=result["resnet101"]["Sign-OPT"][1000],
        SignOPT_resnet101_2000=result["resnet101"]["Sign-OPT"][2000],
        SignOPT_resnet101_5000=result["resnet101"]["Sign-OPT"][5000],
        SignOPT_resnet101_8000=result["resnet101"]["Sign-OPT"][8000],
        SignOPT_resnet101_10000=result["resnet101"]["Sign-OPT"][10000],

        SVMOPT_resnet101_300=result["resnet101"]["SVM-OPT"][300],
        SVMOPT_resnet101_1000=result["resnet101"]["SVM-OPT"][1000],
        SVMOPT_resnet101_2000=result["resnet101"]["SVM-OPT"][2000],
        SVMOPT_resnet101_5000=result["resnet101"]["SVM-OPT"][5000],
        SVMOPT_resnet101_8000=result["resnet101"]["SVM-OPT"][8000],
        SVMOPT_resnet101_10000=result["resnet101"]["SVM-OPT"][10000],

        GeoDA_resnet101_300=result["resnet101"]["GeoDA"][300],
        GeoDA_resnet101_1000=result["resnet101"]["GeoDA"][1000],
        GeoDA_resnet101_2000=result["resnet101"]["GeoDA"][2000],
        GeoDA_resnet101_5000=result["resnet101"]["GeoDA"][5000],
        GeoDA_resnet101_8000=result["resnet101"]["GeoDA"][8000],
        GeoDA_resnet101_10000=result["resnet101"]["GeoDA"][10000],

        RayS_resnet101_300=result["resnet101"]["RayS"][300],
        RayS_resnet101_1000=result["resnet101"]["RayS"][1000],
        RayS_resnet101_2000=result["resnet101"]["RayS"][2000],
        RayS_resnet101_5000=result["resnet101"]["RayS"][5000],
        RayS_resnet101_8000=result["resnet101"]["RayS"][8000],
        RayS_resnet101_10000=result["resnet101"]["RayS"][10000],

        HSJA_resnet101_300=result["resnet101"]["HSJA"][300],
        HSJA_resnet101_1000=result["resnet101"]["HSJA"][1000],
        HSJA_resnet101_2000=result["resnet101"]["HSJA"][2000],
        HSJA_resnet101_5000=result["resnet101"]["HSJA"][5000],
        HSJA_resnet101_8000=result["resnet101"]["HSJA"][8000],
        HSJA_resnet101_10000=result["resnet101"]["HSJA"][10000],

        QEBA_resnet101_300=result["resnet101"]["QEBA"][300],
        QEBA_resnet101_1000=result["resnet101"]["QEBA"][1000],
        QEBA_resnet101_2000=result["resnet101"]["QEBA"][2000],
        QEBA_resnet101_5000=result["resnet101"]["QEBA"][5000],
        QEBA_resnet101_8000=result["resnet101"]["QEBA"][8000],
        QEBA_resnet101_10000=result["resnet101"]["QEBA"][10000],

        TangentAttackHemisphere_resnet101_300=result["resnet101"]["Tangent Attack(hemisphere)"][300],
        TangentAttackHemisphere_resnet101_1000=result["resnet101"]["Tangent Attack(hemisphere)"][1000],
        TangentAttackHemisphere_resnet101_2000=result["resnet101"]["Tangent Attack(hemisphere)"][2000],
        TangentAttackHemisphere_resnet101_5000=result["resnet101"]["Tangent Attack(hemisphere)"][5000],
        TangentAttackHemisphere_resnet101_8000=result["resnet101"]["Tangent Attack(hemisphere)"][8000],
        TangentAttackHemisphere_resnet101_10000=result["resnet101"]["Tangent Attack(hemisphere)"][10000],

        TangentAttackSemiellipsoid_resnet101_300=result["resnet101"]["Tangent Attack(semiellipsoid)"][300],
        TangentAttackSemiellipsoid_resnet101_1000=result["resnet101"]["Tangent Attack(semiellipsoid)"][1000],
        TangentAttackSemiellipsoid_resnet101_2000=result["resnet101"]["Tangent Attack(semiellipsoid)"][2000],
        TangentAttackSemiellipsoid_resnet101_5000=result["resnet101"]["Tangent Attack(semiellipsoid)"][5000],
        TangentAttackSemiellipsoid_resnet101_8000=result["resnet101"]["Tangent Attack(semiellipsoid)"][8000],
        TangentAttackSemiellipsoid_resnet101_10000=result["resnet101"]["Tangent Attack(semiellipsoid)"][10000],

        QEBATangentAttack_resnet101_300=result["resnet101"]["QEBATangentAttack"][300],
        QEBATangentAttack_resnet101_1000=result["resnet101"]["QEBATangentAttack"][1000],
        QEBATangentAttack_resnet101_2000=result["resnet101"]["QEBATangentAttack"][2000],
        QEBATangentAttack_resnet101_5000=result["resnet101"]["QEBATangentAttack"][5000],
        QEBATangentAttack_resnet101_8000=result["resnet101"]["QEBATangentAttack"][8000],
        QEBATangentAttack_resnet101_10000=result["resnet101"]["QEBATangentAttack"][10000],
    )
    )


def draw_table_for_defensive_models_linf_attacks(result):
    print("""
                    Defense | Method  | @300 | @1K | @2K | @5K | @8K | @10K |
                    :- | :-| :- | :- |:- |:- |:- | :- |
                    AT | HSJA | {HSJA_AT_300} | {HSJA_AT_1000} | {HSJA_AT_2000} | {HSJA_AT_5000} | {HSJA_AT_8000} |{HSJA_AT_10000} |
                    AT | Ours(hemishpere) | {TangentAttackHemisphere_AT_300} | {TangentAttackHemisphere_AT_1000} | {TangentAttackHemisphere_AT_2000} | {TangentAttackHemisphere_AT_5000} | {TangentAttackHemisphere_AT_8000} |{TangentAttackHemisphere_AT_10000} |
                    AT | Ours(ellipsoid) | {TangentAttackSemiellipsoid_AT_300} | {TangentAttackSemiellipsoid_AT_1000} | {TangentAttackSemiellipsoid_AT_2000} | {TangentAttackSemiellipsoid_AT_5000} | {TangentAttackSemiellipsoid_AT_8000} |{TangentAttackSemiellipsoid_AT_10000} |
                    TRADES| HSJA | {HSJA_TRADES_300} | {HSJA_TRADES_1000} | {HSJA_TRADES_2000} | {HSJA_TRADES_5000} | {HSJA_TRADES_8000} |{HSJA_TRADES_10000} |
                    TRADES| Ours(hemishpere) | {TangentAttackHemisphere_TRADES_300} | {TangentAttackHemisphere_TRADES_1000} | {TangentAttackHemisphere_TRADES_2000} | {TangentAttackHemisphere_TRADES_5000} | {TangentAttackHemisphere_TRADES_8000} |{TangentAttackHemisphere_TRADES_10000} |
                    TRADES| Ours(ellipsoid) | {TangentAttackSemiellipsoid_TRADES_300} | {TangentAttackSemiellipsoid_TRADES_1000} | {TangentAttackSemiellipsoid_TRADES_2000} | {TangentAttackSemiellipsoid_TRADES_5000} | {TangentAttackSemiellipsoid_TRADES_8000} |{TangentAttackSemiellipsoid_TRADES_10000} |
                    JPEG| HSJA | {HSJA_JPEG_300} | {HSJA_JPEG_1000} | {HSJA_JPEG_2000} | {HSJA_JPEG_5000} | {HSJA_JPEG_8000} |{HSJA_JPEG_10000} |
                    JPEG| Ours(hemishpere) | {TangentAttackHemisphere_JPEG_300} | {TangentAttackHemisphere_JPEG_1000} | {TangentAttackHemisphere_JPEG_2000} | {TangentAttackHemisphere_JPEG_5000} | {TangentAttackHemisphere_JPEG_8000} |{TangentAttackHemisphere_JPEG_10000} |
                    JPEG| Ours(ellipsoid) | {TangentAttackSemiellipsoid_JPEG_300} | {TangentAttackSemiellipsoid_JPEG_1000} | {TangentAttackSemiellipsoid_JPEG_2000} | {TangentAttackSemiellipsoid_JPEG_5000} | {TangentAttackSemiellipsoid_JPEG_8000} |{TangentAttackSemiellipsoid_JPEG_10000} |
                    FeatureDistillation| HSJA | {HSJA_FeatureDistillation_300} | {HSJA_FeatureDistillation_1000} | {HSJA_FeatureDistillation_2000} | {HSJA_FeatureDistillation_5000} | {HSJA_FeatureDistillation_8000} |{HSJA_FeatureDistillation_10000} |
                    FeatureDistillation| Ours(hemishpere) | {TangentAttackHemisphere_FeatureDistillation_300} | {TangentAttackHemisphere_FeatureDistillation_1000} | {TangentAttackHemisphere_FeatureDistillation_2000} | {TangentAttackHemisphere_FeatureDistillation_5000} | {TangentAttackHemisphere_FeatureDistillation_8000} |{TangentAttackHemisphere_FeatureDistillation_10000} |
                    FeatureDistillation| Ours(ellipsoid) | {TangentAttackSemiellipsoid_FeatureDistillation_300} | {TangentAttackSemiellipsoid_FeatureDistillation_1000} | {TangentAttackSemiellipsoid_FeatureDistillation_2000} | {TangentAttackSemiellipsoid_FeatureDistillation_5000} | {TangentAttackSemiellipsoid_FeatureDistillation_8000} |{TangentAttackSemiellipsoid_FeatureDistillation_10000} |
                    Feature Scatter| HSJA | {HSJA_FeatureScatter_300} | {HSJA_FeatureScatter_1000} | {HSJA_FeatureScatter_2000} | {HSJA_FeatureScatter_5000} | {HSJA_FeatureScatter_8000} |{HSJA_FeatureScatter_10000} |
                    Feature Scatter| Ours(hemishpere) | {TangentAttackHemisphere_FeatureScatter_300} | {TangentAttackHemisphere_FeatureScatter_1000} | {TangentAttackHemisphere_FeatureScatter_2000} | {TangentAttackHemisphere_FeatureScatter_5000} | {TangentAttackHemisphere_FeatureScatter_8000} |{TangentAttackHemisphere_FeatureScatter_10000} |
                    Feature Scatter| Ours(ellipsoid) | {TangentAttackSemiellipsoid_FeatureScatter_300} | {TangentAttackSemiellipsoid_FeatureScatter_1000} | {TangentAttackSemiellipsoid_FeatureScatter_2000} | {TangentAttackSemiellipsoid_FeatureScatter_5000} | {TangentAttackSemiellipsoid_FeatureScatter_8000} |{TangentAttackSemiellipsoid_FeatureScatter_10000} |
                                    """.format(
        HSJA_AT_300=result["adv_train"]["HSJA"][300],
        HSJA_AT_1000=result["adv_train"]["HSJA"][1000],
        HSJA_AT_2000=result["adv_train"]["HSJA"][2000],
        HSJA_AT_5000=result["adv_train"]["HSJA"][5000],
        HSJA_AT_8000=result["adv_train"]["HSJA"][8000],
        HSJA_AT_10000=result["adv_train"]["HSJA"][10000],

        TangentAttackHemisphere_AT_300=result["adv_train"]["Tangent Attack(hemisphere)"][300],
        TangentAttackHemisphere_AT_1000=result["adv_train"]["Tangent Attack(hemisphere)"][1000],
        TangentAttackHemisphere_AT_2000=result["adv_train"]["Tangent Attack(hemisphere)"][2000],
        TangentAttackHemisphere_AT_5000=result["adv_train"]["Tangent Attack(hemisphere)"][5000],
        TangentAttackHemisphere_AT_8000=result["adv_train"]["Tangent Attack(hemisphere)"][8000],
        TangentAttackHemisphere_AT_10000=result["adv_train"]["Tangent Attack(hemisphere)"][10000],

        TangentAttackSemiellipsoid_AT_300=result["adv_train"]["Tangent Attack(semiellipsoid)"][300],
        TangentAttackSemiellipsoid_AT_1000=result["adv_train"]["Tangent Attack(semiellipsoid)"][1000],
        TangentAttackSemiellipsoid_AT_2000=result["adv_train"]["Tangent Attack(semiellipsoid)"][2000],
        TangentAttackSemiellipsoid_AT_5000=result["adv_train"]["Tangent Attack(semiellipsoid)"][5000],
        TangentAttackSemiellipsoid_AT_8000=result["adv_train"]["Tangent Attack(semiellipsoid)"][8000],
        TangentAttackSemiellipsoid_AT_10000=result["adv_train"]["Tangent Attack(semiellipsoid)"][10000],

        HSJA_TRADES_300=result["TRADES"]["HSJA"][300],
        HSJA_TRADES_1000=result["TRADES"]["HSJA"][1000],
        HSJA_TRADES_2000=result["TRADES"]["HSJA"][2000],
        HSJA_TRADES_5000=result["TRADES"]["HSJA"][5000],
        HSJA_TRADES_8000=result["TRADES"]["HSJA"][8000],
        HSJA_TRADES_10000=result["TRADES"]["HSJA"][10000],

        TangentAttackHemisphere_TRADES_300=result["TRADES"]["Tangent Attack(hemisphere)"][300],
        TangentAttackHemisphere_TRADES_1000=result["TRADES"]["Tangent Attack(hemisphere)"][1000],
        TangentAttackHemisphere_TRADES_2000=result["TRADES"]["Tangent Attack(hemisphere)"][2000],
        TangentAttackHemisphere_TRADES_5000=result["TRADES"]["Tangent Attack(hemisphere)"][5000],
        TangentAttackHemisphere_TRADES_8000=result["TRADES"]["Tangent Attack(hemisphere)"][8000],
        TangentAttackHemisphere_TRADES_10000=result["TRADES"]["Tangent Attack(hemisphere)"][10000],

        TangentAttackSemiellipsoid_TRADES_300=result["TRADES"]["Tangent Attack(semiellipsoid)"][300],
        TangentAttackSemiellipsoid_TRADES_1000=result["TRADES"]["Tangent Attack(semiellipsoid)"][1000],
        TangentAttackSemiellipsoid_TRADES_2000=result["TRADES"]["Tangent Attack(semiellipsoid)"][2000],
        TangentAttackSemiellipsoid_TRADES_5000=result["TRADES"]["Tangent Attack(semiellipsoid)"][5000],
        TangentAttackSemiellipsoid_TRADES_8000=result["TRADES"]["Tangent Attack(semiellipsoid)"][8000],
        TangentAttackSemiellipsoid_TRADES_10000=result["TRADES"]["Tangent Attack(semiellipsoid)"][10000],

        HSJA_JPEG_300=result["jpeg"]["HSJA"][300],
        HSJA_JPEG_1000=result["jpeg"]["HSJA"][1000],
        HSJA_JPEG_2000=result["jpeg"]["HSJA"][2000],
        HSJA_JPEG_5000=result["jpeg"]["HSJA"][5000],
        HSJA_JPEG_8000=result["jpeg"]["HSJA"][8000],
        HSJA_JPEG_10000=result["jpeg"]["HSJA"][10000],

        TangentAttackHemisphere_JPEG_300=result["jpeg"]["Tangent Attack(hemisphere)"][300],
        TangentAttackHemisphere_JPEG_1000=result["jpeg"]["Tangent Attack(hemisphere)"][1000],
        TangentAttackHemisphere_JPEG_2000=result["jpeg"]["Tangent Attack(hemisphere)"][2000],
        TangentAttackHemisphere_JPEG_5000=result["jpeg"]["Tangent Attack(hemisphere)"][5000],
        TangentAttackHemisphere_JPEG_8000=result["jpeg"]["Tangent Attack(hemisphere)"][8000],
        TangentAttackHemisphere_JPEG_10000=result["jpeg"]["Tangent Attack(hemisphere)"][10000],

        TangentAttackSemiellipsoid_JPEG_300=result["jpeg"]["Tangent Attack(semiellipsoid)"][300],
        TangentAttackSemiellipsoid_JPEG_1000=result["jpeg"]["Tangent Attack(semiellipsoid)"][1000],
        TangentAttackSemiellipsoid_JPEG_2000=result["jpeg"]["Tangent Attack(semiellipsoid)"][2000],
        TangentAttackSemiellipsoid_JPEG_5000=result["jpeg"]["Tangent Attack(semiellipsoid)"][5000],
        TangentAttackSemiellipsoid_JPEG_8000=result["jpeg"]["Tangent Attack(semiellipsoid)"][8000],
        TangentAttackSemiellipsoid_JPEG_10000=result["jpeg"]["Tangent Attack(semiellipsoid)"][10000],

        HSJA_FeatureDistillation_300=result["feature_distillation"]["HSJA"][300],
        HSJA_FeatureDistillation_1000=result["feature_distillation"]["HSJA"][1000],
        HSJA_FeatureDistillation_2000=result["feature_distillation"]["HSJA"][2000],
        HSJA_FeatureDistillation_5000=result["feature_distillation"]["HSJA"][5000],
        HSJA_FeatureDistillation_8000=result["feature_distillation"]["HSJA"][8000],
        HSJA_FeatureDistillation_10000=result["feature_distillation"]["HSJA"][10000],

        TangentAttackHemisphere_FeatureDistillation_300=result["feature_distillation"]["Tangent Attack(hemisphere)"][
            300],
        TangentAttackHemisphere_FeatureDistillation_1000=result["feature_distillation"]["Tangent Attack(hemisphere)"][
            1000],
        TangentAttackHemisphere_FeatureDistillation_2000=result["feature_distillation"]["Tangent Attack(hemisphere)"][
            2000],
        TangentAttackHemisphere_FeatureDistillation_5000=result["feature_distillation"]["Tangent Attack(hemisphere)"][
            5000],
        TangentAttackHemisphere_FeatureDistillation_8000=result["feature_distillation"]["Tangent Attack(hemisphere)"][
            8000],
        TangentAttackHemisphere_FeatureDistillation_10000=result["feature_distillation"]["Tangent Attack(hemisphere)"][
            10000],

        TangentAttackSemiellipsoid_FeatureDistillation_300=
        result["feature_distillation"]["Tangent Attack(semiellipsoid)"][300],
        TangentAttackSemiellipsoid_FeatureDistillation_1000=
        result["feature_distillation"]["Tangent Attack(semiellipsoid)"][1000],
        TangentAttackSemiellipsoid_FeatureDistillation_2000=
        result["feature_distillation"]["Tangent Attack(semiellipsoid)"][2000],
        TangentAttackSemiellipsoid_FeatureDistillation_5000=
        result["feature_distillation"]["Tangent Attack(semiellipsoid)"][5000],
        TangentAttackSemiellipsoid_FeatureDistillation_8000=
        result["feature_distillation"]["Tangent Attack(semiellipsoid)"][8000],
        TangentAttackSemiellipsoid_FeatureDistillation_10000=
        result["feature_distillation"]["Tangent Attack(semiellipsoid)"][10000],

        HSJA_FeatureScatter_300=result["feature_scatter"]["HSJA"][300],
        HSJA_FeatureScatter_1000=result["feature_scatter"]["HSJA"][1000],
        HSJA_FeatureScatter_2000=result["feature_scatter"]["HSJA"][2000],
        HSJA_FeatureScatter_5000=result["feature_scatter"]["HSJA"][5000],
        HSJA_FeatureScatter_8000=result["feature_scatter"]["HSJA"][8000],
        HSJA_FeatureScatter_10000=result["feature_scatter"]["HSJA"][10000],

        TangentAttackHemisphere_FeatureScatter_300=result["feature_scatter"]["Tangent Attack(hemisphere)"][300],
        TangentAttackHemisphere_FeatureScatter_1000=result["feature_scatter"]["Tangent Attack(hemisphere)"][1000],
        TangentAttackHemisphere_FeatureScatter_2000=result["feature_scatter"]["Tangent Attack(hemisphere)"][2000],
        TangentAttackHemisphere_FeatureScatter_5000=result["feature_scatter"]["Tangent Attack(hemisphere)"][5000],
        TangentAttackHemisphere_FeatureScatter_8000=result["feature_scatter"]["Tangent Attack(hemisphere)"][8000],
        TangentAttackHemisphere_FeatureScatter_10000=result["feature_scatter"]["Tangent Attack(hemisphere)"][10000],

        TangentAttackSemiellipsoid_FeatureScatter_300=result["feature_scatter"]["Tangent Attack(semiellipsoid)"][300],
        TangentAttackSemiellipsoid_FeatureScatter_1000=result["feature_scatter"]["Tangent Attack(semiellipsoid)"][1000],
        TangentAttackSemiellipsoid_FeatureScatter_2000=result["feature_scatter"]["Tangent Attack(semiellipsoid)"][2000],
        TangentAttackSemiellipsoid_FeatureScatter_5000=result["feature_scatter"]["Tangent Attack(semiellipsoid)"][5000],
        TangentAttackSemiellipsoid_FeatureScatter_8000=result["feature_scatter"]["Tangent Attack(semiellipsoid)"][8000],
        TangentAttackSemiellipsoid_FeatureScatter_10000=result["feature_scatter"]["Tangent Attack(semiellipsoid)"][
            10000],
    )
    )


def draw_appendix_table_for_defensive_models_linf_attacks(result):
    print("""
                AT  & Sign-OPT & {SignOPT_AT_300} &  {SignOPT_AT_1000} &  {SignOPT_AT_2000} &  {SignOPT_AT_5000} &  {SignOPT_AT_8000} & {SignOPT_AT_10000} \\\\
                    & SVM-OPT & {SVMOPT_AT_300} &  {SVMOPT_AT_1000} &  {SVMOPT_AT_2000} &  {SVMOPT_AT_5000} &  {SVMOPT_AT_8000} & {SVMOPT_AT_10000} \\\\
                    &  HSJA &  {HSJA_AT_300} &  {HSJA_AT_1000} &  {HSJA_AT_2000} &  {HSJA_AT_5000} &  {HSJA_AT_8000} & {HSJA_AT_10000} \\\\
                    &  Ours(hemishpere) &  {TangentAttackHemisphere_AT_300} &  {TangentAttackHemisphere_AT_1000} &  {TangentAttackHemisphere_AT_2000} &  {TangentAttackHemisphere_AT_5000} &  {TangentAttackHemisphere_AT_8000} & {TangentAttackHemisphere_AT_10000} \\\\
                    &  Ours(ellipsoid) &  {TangentAttackSemiellipsoid_AT_300} &  {TangentAttackSemiellipsoid_AT_1000} &  {TangentAttackSemiellipsoid_AT_2000} &  {TangentAttackSemiellipsoid_AT_5000} &  {TangentAttackSemiellipsoid_AT_8000} & {TangentAttackSemiellipsoid_AT_10000} \\\\
                TRADES & Sign-OPT & {SignOPT_TRADES_300} &  {SignOPT_TRADES_1000} &  {SignOPT_TRADES_2000} &  {SignOPT_TRADES_5000} &  {SignOPT_TRADES_8000} & {SignOPT_TRADES_10000} \\\\
                    & SVM-OPT & {SVMOPT_TRADES_300} &  {SVMOPT_TRADES_1000} &  {SVMOPT_TRADES_2000} &  {SVMOPT_TRADES_5000} &  {SVMOPT_TRADES_8000} & {SVMOPT_TRADES_10000} \\\\
                    &  HSJA &  {HSJA_TRADES_300} &  {HSJA_TRADES_1000} &  {HSJA_TRADES_2000} &  {HSJA_TRADES_5000} &  {HSJA_TRADES_8000} & {HSJA_TRADES_10000} \\\\
                    &  Ours(hemishpere) &  {TangentAttackHemisphere_TRADES_300} &  {TangentAttackHemisphere_TRADES_1000} &  {TangentAttackHemisphere_TRADES_2000} &  {TangentAttackHemisphere_TRADES_5000} &  {TangentAttackHemisphere_TRADES_8000} & {TangentAttackHemisphere_TRADES_10000}  \\\\
                    &  Ours(ellipsoid) &  {TangentAttackSemiellipsoid_TRADES_300} &  {TangentAttackSemiellipsoid_TRADES_1000} &  {TangentAttackSemiellipsoid_TRADES_2000} &  {TangentAttackSemiellipsoid_TRADES_5000} &  {TangentAttackSemiellipsoid_TRADES_8000} & {TangentAttackSemiellipsoid_TRADES_10000} \\\\
                JPEG& Sign-OPT & {SignOPT_JPEG_300} &  {SignOPT_JPEG_1000} &  {SignOPT_JPEG_2000} &  {SignOPT_JPEG_5000} &  {SignOPT_JPEG_8000} & {SignOPT_JPEG_10000} \\\\
                    & SVM-OPT & {SVMOPT_JPEG_300} &  {SVMOPT_JPEG_1000} &  {SVMOPT_JPEG_2000} &  {SVMOPT_JPEG_5000} &  {SVMOPT_JPEG_8000} & {SVMOPT_JPEG_10000} \\\\
                    & HSJA &  {HSJA_JPEG_300} &  {HSJA_JPEG_1000} &  {HSJA_JPEG_2000} &  {HSJA_JPEG_5000} &  {HSJA_JPEG_8000} & {HSJA_JPEG_10000} \\\\
                    &  Ours(hemishpere) &  {TangentAttackHemisphere_JPEG_300} &  {TangentAttackHemisphere_JPEG_1000} &  {TangentAttackHemisphere_JPEG_2000} &  {TangentAttackHemisphere_JPEG_5000} &  {TangentAttackHemisphere_JPEG_8000} & {TangentAttackHemisphere_JPEG_10000} \\\\
                    &  Ours(ellipsoid) &  {TangentAttackSemiellipsoid_JPEG_300} &  {TangentAttackSemiellipsoid_JPEG_1000} &  {TangentAttackSemiellipsoid_JPEG_2000} &  {TangentAttackSemiellipsoid_JPEG_5000} &  {TangentAttackSemiellipsoid_JPEG_8000} & {TangentAttackSemiellipsoid_JPEG_10000} \\\\
               Feature Distillation&  Sign-OPT & {SignOPT_FeatureDistillation_300} &  {SignOPT_FeatureDistillation_1000} &  {SignOPT_FeatureDistillation_2000} &  {SignOPT_FeatureDistillation_5000} &  {SignOPT_FeatureDistillation_8000} & {SignOPT_FeatureDistillation_10000} \\\\
                    & SVM-OPT & {SVMOPT_FeatureDistillation_300} &  {SVMOPT_FeatureDistillation_1000} &  {SVMOPT_FeatureDistillation_2000} &  {SVMOPT_FeatureDistillation_5000} &  {SVMOPT_FeatureDistillation_8000} & {SVMOPT_FeatureDistillation_10000} \\\\
                    & HSJA &  {HSJA_FeatureDistillation_300} &  {HSJA_FeatureDistillation_1000} &  {HSJA_FeatureDistillation_2000} &  {HSJA_FeatureDistillation_5000} &  {HSJA_FeatureDistillation_8000} & {HSJA_FeatureDistillation_10000} \\\\
                    &  Ours(hemishpere) &  {TangentAttackHemisphere_FeatureDistillation_300} &  {TangentAttackHemisphere_FeatureDistillation_1000} &  {TangentAttackHemisphere_FeatureDistillation_2000} &  {TangentAttackHemisphere_FeatureDistillation_5000} &  {TangentAttackHemisphere_FeatureDistillation_8000} & {TangentAttackHemisphere_FeatureDistillation_10000} \\\\
                    &  Ours(ellipsoid) &  {TangentAttackSemiellipsoid_FeatureDistillation_300} &  {TangentAttackSemiellipsoid_FeatureDistillation_1000} &  {TangentAttackSemiellipsoid_FeatureDistillation_2000} &  {TangentAttackSemiellipsoid_FeatureDistillation_5000} &  {TangentAttackSemiellipsoid_FeatureDistillation_8000} & {TangentAttackSemiellipsoid_FeatureDistillation_10000} \\\\
               Feature Scatter & Sign-OPT & {SignOPT_FeatureScatter_300} &  {SignOPT_FeatureScatter_1000} &  {SignOPT_FeatureScatter_2000} &  {SignOPT_FeatureScatter_5000} &  {SignOPT_FeatureScatter_8000} & {SignOPT_FeatureScatter_10000} \\\\
                    & SVM-OPT & {SVMOPT_FeatureScatter_300} &  {SVMOPT_FeatureScatter_1000} &  {SVMOPT_FeatureScatter_2000} &  {SVMOPT_FeatureScatter_5000} &  {SVMOPT_FeatureScatter_8000} & {SVMOPT_FeatureScatter_10000} \\\\
                    &  HSJA &  {HSJA_FeatureScatter_300} &  {HSJA_FeatureScatter_1000} &  {HSJA_FeatureScatter_2000} &  {HSJA_FeatureScatter_5000} &  {HSJA_FeatureScatter_8000} & {HSJA_FeatureScatter_10000} \\\\
                    &  Ours(hemishpere) &  {TangentAttackHemisphere_FeatureScatter_300} &  {TangentAttackHemisphere_FeatureScatter_1000} &  {TangentAttackHemisphere_FeatureScatter_2000} &  {TangentAttackHemisphere_FeatureScatter_5000} &  {TangentAttackHemisphere_FeatureScatter_8000} & {TangentAttackHemisphere_FeatureScatter_10000} \\\\
                    &  Ours(ellipsoid) &  {TangentAttackSemiellipsoid_FeatureScatter_300} &  {TangentAttackSemiellipsoid_FeatureScatter_1000} &  {TangentAttackSemiellipsoid_FeatureScatter_2000} &  {TangentAttackSemiellipsoid_FeatureScatter_5000} &  {TangentAttackSemiellipsoid_FeatureScatter_8000} & {TangentAttackSemiellipsoid_FeatureScatter_10000} \\\\
                                    """.format(
        SVMOPT_AT_300=result["adv_train"]["SVM-OPT"][300],
        SVMOPT_AT_1000=result["adv_train"]["SVM-OPT"][1000],
        SVMOPT_AT_2000=result["adv_train"]["SVM-OPT"][2000],
        SVMOPT_AT_5000=result["adv_train"]["SVM-OPT"][5000],
        SVMOPT_AT_8000=result["adv_train"]["SVM-OPT"][8000],
        SVMOPT_AT_10000=result["adv_train"]["SVM-OPT"][10000],

        SignOPT_AT_300=result["adv_train"]["Sign-OPT"][300],
        SignOPT_AT_1000=result["adv_train"]["Sign-OPT"][1000],
        SignOPT_AT_2000=result["adv_train"]["Sign-OPT"][2000],
        SignOPT_AT_5000=result["adv_train"]["Sign-OPT"][5000],
        SignOPT_AT_8000=result["adv_train"]["Sign-OPT"][8000],
        SignOPT_AT_10000=result["adv_train"]["Sign-OPT"][10000],

        SVMOPT_TRADES_300=result["adv_train"]["SVM-OPT"][300],
        SVMOPT_TRADES_1000=result["adv_train"]["SVM-OPT"][1000],
        SVMOPT_TRADES_2000=result["adv_train"]["SVM-OPT"][2000],
        SVMOPT_TRADES_5000=result["adv_train"]["SVM-OPT"][5000],
        SVMOPT_TRADES_8000=result["adv_train"]["SVM-OPT"][8000],
        SVMOPT_TRADES_10000=result["adv_train"]["SVM-OPT"][10000],

        SignOPT_TRADES_300=result["adv_train"]["Sign-OPT"][300],
        SignOPT_TRADES_1000=result["adv_train"]["Sign-OPT"][1000],
        SignOPT_TRADES_2000=result["adv_train"]["Sign-OPT"][2000],
        SignOPT_TRADES_5000=result["adv_train"]["Sign-OPT"][5000],
        SignOPT_TRADES_8000=result["adv_train"]["Sign-OPT"][8000],
        SignOPT_TRADES_10000=result["adv_train"]["Sign-OPT"][10000],

        SVMOPT_JPEG_300=result["adv_train"]["SVM-OPT"][300],
        SVMOPT_JPEG_1000=result["adv_train"]["SVM-OPT"][1000],
        SVMOPT_JPEG_2000=result["adv_train"]["SVM-OPT"][2000],
        SVMOPT_JPEG_5000=result["adv_train"]["SVM-OPT"][5000],
        SVMOPT_JPEG_8000=result["adv_train"]["SVM-OPT"][8000],
        SVMOPT_JPEG_10000=result["adv_train"]["SVM-OPT"][10000],

        SignOPT_JPEG_300=result["adv_train"]["Sign-OPT"][300],
        SignOPT_JPEG_1000=result["adv_train"]["Sign-OPT"][1000],
        SignOPT_JPEG_2000=result["adv_train"]["Sign-OPT"][2000],
        SignOPT_JPEG_5000=result["adv_train"]["Sign-OPT"][5000],
        SignOPT_JPEG_8000=result["adv_train"]["Sign-OPT"][8000],
        SignOPT_JPEG_10000=result["adv_train"]["Sign-OPT"][10000],

        SVMOPT_FeatureDistillation_300=result["adv_train"]["SVM-OPT"][300],
        SVMOPT_FeatureDistillation_1000=result["adv_train"]["SVM-OPT"][1000],
        SVMOPT_FeatureDistillation_2000=result["adv_train"]["SVM-OPT"][2000],
        SVMOPT_FeatureDistillation_5000=result["adv_train"]["SVM-OPT"][5000],
        SVMOPT_FeatureDistillation_8000=result["adv_train"]["SVM-OPT"][8000],
        SVMOPT_FeatureDistillation_10000=result["adv_train"]["SVM-OPT"][10000],

        SignOPT_FeatureDistillation_300=result["adv_train"]["Sign-OPT"][300],
        SignOPT_FeatureDistillation_1000=result["adv_train"]["Sign-OPT"][1000],
        SignOPT_FeatureDistillation_2000=result["adv_train"]["Sign-OPT"][2000],
        SignOPT_FeatureDistillation_5000=result["adv_train"]["Sign-OPT"][5000],
        SignOPT_FeatureDistillation_8000=result["adv_train"]["Sign-OPT"][8000],
        SignOPT_FeatureDistillation_10000=result["adv_train"]["Sign-OPT"][10000],

        SVMOPT_FeatureScatter_300=result["adv_train"]["SVM-OPT"][300],
        SVMOPT_FeatureScatter_1000=result["adv_train"]["SVM-OPT"][1000],
        SVMOPT_FeatureScatter_2000=result["adv_train"]["SVM-OPT"][2000],
        SVMOPT_FeatureScatter_5000=result["adv_train"]["SVM-OPT"][5000],
        SVMOPT_FeatureScatter_8000=result["adv_train"]["SVM-OPT"][8000],
        SVMOPT_FeatureScatter_10000=result["adv_train"]["SVM-OPT"][10000],

        SignOPT_FeatureScatter_300=result["adv_train"]["Sign-OPT"][300],
        SignOPT_FeatureScatter_1000=result["adv_train"]["Sign-OPT"][1000],
        SignOPT_FeatureScatter_2000=result["adv_train"]["Sign-OPT"][2000],
        SignOPT_FeatureScatter_5000=result["adv_train"]["Sign-OPT"][5000],
        SignOPT_FeatureScatter_8000=result["adv_train"]["Sign-OPT"][8000],
        SignOPT_FeatureScatter_10000=result["adv_train"]["Sign-OPT"][10000],

        HSJA_AT_300=result["adv_train"]["HSJA"][300],
        HSJA_AT_1000=result["adv_train"]["HSJA"][1000],
        HSJA_AT_2000=result["adv_train"]["HSJA"][2000],
        HSJA_AT_5000=result["adv_train"]["HSJA"][5000],
        HSJA_AT_8000=result["adv_train"]["HSJA"][8000],
        HSJA_AT_10000=result["adv_train"]["HSJA"][10000],

        TangentAttackHemisphere_AT_300=result["adv_train"]["Tangent Attack(hemisphere)"][300],
        TangentAttackHemisphere_AT_1000=result["adv_train"]["Tangent Attack(hemisphere)"][1000],
        TangentAttackHemisphere_AT_2000=result["adv_train"]["Tangent Attack(hemisphere)"][2000],
        TangentAttackHemisphere_AT_5000=result["adv_train"]["Tangent Attack(hemisphere)"][5000],
        TangentAttackHemisphere_AT_8000=result["adv_train"]["Tangent Attack(hemisphere)"][8000],
        TangentAttackHemisphere_AT_10000=result["adv_train"]["Tangent Attack(hemisphere)"][10000],

        TangentAttackSemiellipsoid_AT_300=result["adv_train"]["Tangent Attack(semiellipsoid)"][300],
        TangentAttackSemiellipsoid_AT_1000=result["adv_train"]["Tangent Attack(semiellipsoid)"][1000],
        TangentAttackSemiellipsoid_AT_2000=result["adv_train"]["Tangent Attack(semiellipsoid)"][2000],
        TangentAttackSemiellipsoid_AT_5000=result["adv_train"]["Tangent Attack(semiellipsoid)"][5000],
        TangentAttackSemiellipsoid_AT_8000=result["adv_train"]["Tangent Attack(semiellipsoid)"][8000],
        TangentAttackSemiellipsoid_AT_10000=result["adv_train"]["Tangent Attack(semiellipsoid)"][10000],

        HSJA_TRADES_300=result["TRADES"]["HSJA"][300],
        HSJA_TRADES_1000=result["TRADES"]["HSJA"][1000],
        HSJA_TRADES_2000=result["TRADES"]["HSJA"][2000],
        HSJA_TRADES_5000=result["TRADES"]["HSJA"][5000],
        HSJA_TRADES_8000=result["TRADES"]["HSJA"][8000],
        HSJA_TRADES_10000=result["TRADES"]["HSJA"][10000],

        TangentAttackHemisphere_TRADES_300=result["TRADES"]["Tangent Attack(hemisphere)"][300],
        TangentAttackHemisphere_TRADES_1000=result["TRADES"]["Tangent Attack(hemisphere)"][1000],
        TangentAttackHemisphere_TRADES_2000=result["TRADES"]["Tangent Attack(hemisphere)"][2000],
        TangentAttackHemisphere_TRADES_5000=result["TRADES"]["Tangent Attack(hemisphere)"][5000],
        TangentAttackHemisphere_TRADES_8000=result["TRADES"]["Tangent Attack(hemisphere)"][8000],
        TangentAttackHemisphere_TRADES_10000=result["TRADES"]["Tangent Attack(hemisphere)"][10000],

        TangentAttackSemiellipsoid_TRADES_300=result["TRADES"]["Tangent Attack(semiellipsoid)"][300],
        TangentAttackSemiellipsoid_TRADES_1000=result["TRADES"]["Tangent Attack(semiellipsoid)"][1000],
        TangentAttackSemiellipsoid_TRADES_2000=result["TRADES"]["Tangent Attack(semiellipsoid)"][2000],
        TangentAttackSemiellipsoid_TRADES_5000=result["TRADES"]["Tangent Attack(semiellipsoid)"][5000],
        TangentAttackSemiellipsoid_TRADES_8000=result["TRADES"]["Tangent Attack(semiellipsoid)"][8000],
        TangentAttackSemiellipsoid_TRADES_10000=result["TRADES"]["Tangent Attack(semiellipsoid)"][10000],

        HSJA_JPEG_300=result["jpeg"]["HSJA"][300],
        HSJA_JPEG_1000=result["jpeg"]["HSJA"][1000],
        HSJA_JPEG_2000=result["jpeg"]["HSJA"][2000],
        HSJA_JPEG_5000=result["jpeg"]["HSJA"][5000],
        HSJA_JPEG_8000=result["jpeg"]["HSJA"][8000],
        HSJA_JPEG_10000=result["jpeg"]["HSJA"][10000],

        TangentAttackHemisphere_JPEG_300=result["jpeg"]["Tangent Attack(hemisphere)"][300],
        TangentAttackHemisphere_JPEG_1000=result["jpeg"]["Tangent Attack(hemisphere)"][1000],
        TangentAttackHemisphere_JPEG_2000=result["jpeg"]["Tangent Attack(hemisphere)"][2000],
        TangentAttackHemisphere_JPEG_5000=result["jpeg"]["Tangent Attack(hemisphere)"][5000],
        TangentAttackHemisphere_JPEG_8000=result["jpeg"]["Tangent Attack(hemisphere)"][8000],
        TangentAttackHemisphere_JPEG_10000=result["jpeg"]["Tangent Attack(hemisphere)"][10000],

        TangentAttackSemiellipsoid_JPEG_300=result["jpeg"]["Tangent Attack(semiellipsoid)"][300],
        TangentAttackSemiellipsoid_JPEG_1000=result["jpeg"]["Tangent Attack(semiellipsoid)"][1000],
        TangentAttackSemiellipsoid_JPEG_2000=result["jpeg"]["Tangent Attack(semiellipsoid)"][2000],
        TangentAttackSemiellipsoid_JPEG_5000=result["jpeg"]["Tangent Attack(semiellipsoid)"][5000],
        TangentAttackSemiellipsoid_JPEG_8000=result["jpeg"]["Tangent Attack(semiellipsoid)"][8000],
        TangentAttackSemiellipsoid_JPEG_10000=result["jpeg"]["Tangent Attack(semiellipsoid)"][10000],

        HSJA_FeatureDistillation_300=result["feature_distillation"]["HSJA"][300],
        HSJA_FeatureDistillation_1000=result["feature_distillation"]["HSJA"][1000],
        HSJA_FeatureDistillation_2000=result["feature_distillation"]["HSJA"][2000],
        HSJA_FeatureDistillation_5000=result["feature_distillation"]["HSJA"][5000],
        HSJA_FeatureDistillation_8000=result["feature_distillation"]["HSJA"][8000],
        HSJA_FeatureDistillation_10000=result["feature_distillation"]["HSJA"][10000],

        TangentAttackHemisphere_FeatureDistillation_300=result["feature_distillation"]["Tangent Attack(hemisphere)"][
            300],
        TangentAttackHemisphere_FeatureDistillation_1000=result["feature_distillation"]["Tangent Attack(hemisphere)"][
            1000],
        TangentAttackHemisphere_FeatureDistillation_2000=result["feature_distillation"]["Tangent Attack(hemisphere)"][
            2000],
        TangentAttackHemisphere_FeatureDistillation_5000=result["feature_distillation"]["Tangent Attack(hemisphere)"][
            5000],
        TangentAttackHemisphere_FeatureDistillation_8000=result["feature_distillation"]["Tangent Attack(hemisphere)"][
            8000],
        TangentAttackHemisphere_FeatureDistillation_10000=result["feature_distillation"]["Tangent Attack(hemisphere)"][
            10000],

        TangentAttackSemiellipsoid_FeatureDistillation_300=
        result["feature_distillation"]["Tangent Attack(semiellipsoid)"][300],
        TangentAttackSemiellipsoid_FeatureDistillation_1000=
        result["feature_distillation"]["Tangent Attack(semiellipsoid)"][1000],
        TangentAttackSemiellipsoid_FeatureDistillation_2000=
        result["feature_distillation"]["Tangent Attack(semiellipsoid)"][2000],
        TangentAttackSemiellipsoid_FeatureDistillation_5000=
        result["feature_distillation"]["Tangent Attack(semiellipsoid)"][5000],
        TangentAttackSemiellipsoid_FeatureDistillation_8000=
        result["feature_distillation"]["Tangent Attack(semiellipsoid)"][8000],
        TangentAttackSemiellipsoid_FeatureDistillation_10000=
        result["feature_distillation"]["Tangent Attack(semiellipsoid)"][10000],

        HSJA_FeatureScatter_300=result["feature_scatter"]["HSJA"][300],
        HSJA_FeatureScatter_1000=result["feature_scatter"]["HSJA"][1000],
        HSJA_FeatureScatter_2000=result["feature_scatter"]["HSJA"][2000],
        HSJA_FeatureScatter_5000=result["feature_scatter"]["HSJA"][5000],
        HSJA_FeatureScatter_8000=result["feature_scatter"]["HSJA"][8000],
        HSJA_FeatureScatter_10000=result["feature_scatter"]["HSJA"][10000],

        TangentAttackHemisphere_FeatureScatter_300=result["feature_scatter"]["Tangent Attack(hemisphere)"][300],
        TangentAttackHemisphere_FeatureScatter_1000=result["feature_scatter"]["Tangent Attack(hemisphere)"][1000],
        TangentAttackHemisphere_FeatureScatter_2000=result["feature_scatter"]["Tangent Attack(hemisphere)"][2000],
        TangentAttackHemisphere_FeatureScatter_5000=result["feature_scatter"]["Tangent Attack(hemisphere)"][5000],
        TangentAttackHemisphere_FeatureScatter_8000=result["feature_scatter"]["Tangent Attack(hemisphere)"][8000],
        TangentAttackHemisphere_FeatureScatter_10000=result["feature_scatter"]["Tangent Attack(hemisphere)"][10000],

        TangentAttackSemiellipsoid_FeatureScatter_300=result["feature_scatter"]["Tangent Attack(semiellipsoid)"][300],
        TangentAttackSemiellipsoid_FeatureScatter_1000=result["feature_scatter"]["Tangent Attack(semiellipsoid)"][1000],
        TangentAttackSemiellipsoid_FeatureScatter_2000=result["feature_scatter"]["Tangent Attack(semiellipsoid)"][2000],
        TangentAttackSemiellipsoid_FeatureScatter_5000=result["feature_scatter"]["Tangent Attack(semiellipsoid)"][5000],
        TangentAttackSemiellipsoid_FeatureScatter_8000=result["feature_scatter"]["Tangent Attack(semiellipsoid)"][8000],
        TangentAttackSemiellipsoid_FeatureScatter_10000=result["feature_scatter"]["Tangent Attack(semiellipsoid)"][
            10000],
    )
    )



defensive_method_name_to_paper = {"tangent_attack":"Tangent Attack(hemisphere)",  "HSJA":"HSJA",
                        "ellipsoid_tangent_attack":"Tangent Attack(semiellipsoid)",
                        "SignOPT":"Sign-OPT", "SVMOPT":"SVM-OPT"
                        }

def from_method_to_defensive_path(dataset, method, norm, targeted,target_type):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if method == "tangent_attack":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm, target_str=target_str)
    elif method == "ellipsoid_tangent_attack":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                               norm=norm,
                                                               target_str=target_str)
    elif method == "HSJA":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm,  target_str=target_str)
    elif method == "SignOPT":
        if targeted:
            path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                                   target_str="untargeted" if not targeted else "targeted_increment")
        else:
            path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SVMOPT":
        if targeted:
            path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset, norm=norm,
                                                                   target_str="untargeted" if not targeted else "targeted_increment")
        else:
            path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method,dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment")

    return path


def get_defensive_file_name_list(dataset, method_name_to_paper, norm, targeted, target_type):
    folder_path_dict = {}
    for method, paper_method_name in method_name_to_paper.items():
        file_path = "/home1/machen/hard_label_attacks/logs/" + from_method_to_defensive_path(dataset, method, norm, targeted,target_type)
        folder_path_dict[paper_method_name] = file_path
    return folder_path_dict

def fetch_defensive_json_content_given_contraint(dataset, norm, targeted, target_type, arch, defensive_methods, query_budgets, want_key="mean_distortion"):
    folder_list = get_defensive_file_name_list(dataset, defensive_method_name_to_paper, norm, targeted,target_type)
    result = defaultdict(dict)
    for method, folder in folder_list.items():
        for defense in defensive_methods:
            file_path = folder + "/{}_{}_result.json".format(arch,defense)
            if method in ["RayS","GeoDA"] and targeted:
                print("{} does not exist!".format(file_path))
                result[method] = defaultdict(lambda : "-")
                continue
            if not os.path.exists(file_path):
                distortion_dict = {}
            else:
                distortion_dict = read_json_and_extract(file_path)
            print(file_path)
            mean_and_median_distortions = get_mean_and_median_distortion_given_query_budgets(distortion_dict, query_budgets,want_key)
            result[defense][method] = mean_and_median_distortions
    return result



if __name__ == "__main__":
    dataset = "CIFAR-10"
    norm = "linf"
    if "CIFAR" in dataset:
        archs = ['pyramidnet272',"gdas","WRN-28-10-drop", "WRN-40-10-drop"]
    else:
        archs = ["resnet101"]
    query_budgets = [300,1000,2000,5000,8000,10000]
    targeted= False
    target_type = "increment"
    # if targeted:
    #     query_budgets.extend([12000,15000,18000,20000])


    # if "CIFAR" in dataset:
    #     targeted_result = {}
    #     for arch in archs:
    #         result = fetch_all_json_content_given_contraint(dataset, norm, True,target_type, arch, query_budgets, "mean_distortion")
    #         targeted_result[arch] = result
    #     untargeted_result = {}
    #     for arch in archs:
    #         result = fetch_all_json_content_given_contraint(dataset, norm, False,target_type, arch, query_budgets, "mean_distortion")
    #         untargeted_result[arch] = result
    #
    # else:
    #     result_archs = {}
    #     for arch in archs:
    #         result = fetch_all_json_content_given_contraint(dataset, norm, targeted,target_type, arch, query_budgets, "mean_distortion")
    #         result_archs[arch] = result
    #
    #     draw_tables_for_ImageNet_10K_to_20K(result_archs)

    # print("--------------below is defensive model for linf norm attacks on CIFAR-10------------------------")
    defensive_result = fetch_defensive_json_content_given_contraint(dataset, norm, targeted, target_type, "resnet-50",
                                                                    ["adv_train","jpeg","feature_scatter","TRADES","feature_distillation"], query_budgets, "mean_distortion")
    draw_appendix_table_for_defensive_models_linf_attacks(defensive_result)
