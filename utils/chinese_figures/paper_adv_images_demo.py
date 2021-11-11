import argparse
import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import PIL.Image
from torchvision.transforms import transforms

import boundary_attack.foolbox as foolbox
from QEBA.adversarial import Adversarial
from QEBA.attack import QEBA
from QEBA.rv_generator import load_pgen
from QEBA.utils import TargetClass, Misclassification
from QEBATangentAttack.attack import QEBATangentAttack
from SignOPT.sign_opt_l2_norm_attack import SignOptL2Norm
from SignOPT.sign_opt_linf_norm_attack import SignOptLinfNorm
from boundary_attack.foolbox.attacks import BoundaryAttack
from config import  CLASS_NUM, IN_CHANNELS, IMAGE_DATA_ROOT
from dataset.target_class_dataset import ImageNetDataset, TinyImageNetDataset, CIFAR10Dataset, CIFAR100Dataset
from hop_skip_jump_attack.attack import HotSkipJumpAttack
from models.standard_model import StandardModel
from tangent_attack_hemisphere.attack import TangentAttack
from tangent_attack_semiellipsoid.attack import EllipsoidTangentAttack
from torch.nn import functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Drawing Figures of Hard-label Attacks')
    parser.add_argument("--attack", type=str, choices=["HSJA","BA","TangentAttack","GeneralizedTangentAttack",
                                                         "QEBA","QEBATangentAttack","SignOPT","SVMOPT"])
    parser.add_argument("--dataset", type=str, choices=["CIFAR-10","ImageNet","CIFAR-100"],required=True, help="the dataset to train")
    parser.add_argument("--norm", type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument("--targeted", action="store_true", help="Does it train on the data of targeted attack?")
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument('--max_queries',type=int,default=10000)
    args = parser.parse_args()
    return args

def construct_attack(max_queries, model, dataset, target_label, attack, targeted, norm, epsilon):
    maxN = 10000
    initN = 100
    ITER = 150
    if dataset.startswith("CIFAR") or dataset=="TinyImageNet":
        if norm == "l2":
            gamma = 1.0
        else:
            gamma = 100.0
    elif dataset == "ImageNet":
        if norm == "l2":
            gamma = 1000.0
        else:
            gamma = 10000.0

    if attack == "HSJA":
        attacker = HotSkipJumpAttack(model, dataset, clip_min=0,clip_max=1.0,height=model.input_size[-2],
                                     width=model.input_size[-1], channels=IN_CHANNELS[dataset],norm=norm,
                                     epsilon=epsilon,gamma=gamma,maximum_queries=max_queries)
    elif attack == "BA":
        if not targeted:
            criterion = foolbox.criteria.Misclassification()
        else:
            criterion = foolbox.criteria.TargetClass(target_label.item())
        fmodel= foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=CLASS_NUM[dataset],
                                            device=str(torch.device('cuda')))
        attacker = BoundaryAttack(fmodel,criterion)
    elif attack == "TangentAttack":

        attacker = TangentAttack(model, dataset, 0, 1.0, model.input_size[-2], model.input_size[-1],
                                 IN_CHANNELS[dataset],
                                 norm, epsilon, 10000, gamma=gamma,
                                 stepsize_search="geometric_progression",
                                 max_num_evals=1e4,
                                 init_num_evals=100, maximum_queries=max_queries,
                                 verify_tangent_point=False)
    elif attack == "GeneralizedTangentAttack":
        attacker = EllipsoidTangentAttack(model, dataset, 0, 1.0, model.input_size[-2], model.input_size[-1],
                                 IN_CHANNELS[dataset],
                                 norm, epsilon, 10000, gamma=gamma,
                                 stepsize_search="geometric_progression",
                                 max_num_evals=1e4,
                                 init_num_evals=100, maximum_queries=max_queries,
                                 verify_tangent_point=False)
    elif attack == "QEBA":
        PGEN = "resize"
        p_gen = load_pgen(dataset, PGEN, None)
        attacker = QEBA(model, dataset, 0, 1.0, model.input_size[-2], model.input_size[-1], IN_CHANNELS[dataset],
                        norm, epsilon, iterations=ITER, initial_num_evals=initN, max_num_evals=maxN,
                       internal_dtype=torch.float32,rv_generator=p_gen, atk_level=999, mask=None,
                        gamma=0.01, batch_size=256, stepsize_search="geometric_progression",
                        log_every_n_steps=1, suffix=PGEN, verbose=False, maximum_queries=max_queries)

    elif attack == "QEBATangentAttack":
        PGEN = "resize"
        p_gen = load_pgen(dataset, PGEN, None)
        attacker = QEBATangentAttack(model, dataset, 0, 1.0, model.input_size[-2], model.input_size[-1], IN_CHANNELS[dataset],
                        norm, epsilon, iterations=ITER, initial_num_evals=initN, max_num_evals=maxN,
                       internal_dtype=torch.float32,rv_generator=p_gen, atk_level=999, mask=None,
                        gamma=0.01, batch_size=256, stepsize_search="geometric_progression",
                        log_every_n_steps=1, suffix=PGEN, verbose=False, maximum_queries=max_queries)
    elif attack == "SignOPT":
        if norm == "l2":
            attacker = SignOptL2Norm(model, dataset, epsilon, targeted,
                                     1, 100,
                                    maximum_queries=max_queries,svm=False,tot=1e-4,
                                     best_initial_target_sample=False)
        elif norm == "linf":
            attacker = SignOptLinfNorm(model, dataset, epsilon, targeted,
                                     1, 100,
                                    maximum_queries=max_queries,svm=False,tot=None,
                                     best_initial_target_sample=False)
    elif attack == "SVMOPT":
        if norm == "l2":
            attacker = SignOptL2Norm(model, dataset, epsilon, targeted,
                                     1, 100,
                                    maximum_queries=max_queries,svm=True,tot=1e-4,
                                     best_initial_target_sample=False)
        elif norm == "linf":
            attacker = SignOptLinfNorm(model, dataset, epsilon, targeted,
                                     1, 100,
                                    maximum_queries=max_queries,svm=True,tot=None,
                                     best_initial_target_sample=False)
    return attacker

def get_image_of_target_class(dataset_name, target_labels, target_model):

    images = []
    for label in target_labels:  # length of target_labels is 1
        if dataset_name == "ImageNet":
            dataset = ImageNetDataset(IMAGE_DATA_ROOT[dataset_name],label.item(), "validation")
        elif dataset_name == "TinyImageNet":
            dataset = TinyImageNetDataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
        elif dataset_name == "CIFAR-10":
            dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
        elif dataset_name=="CIFAR-100":
            dataset = CIFAR100Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")

        index = len(dataset)//2
        image, true_label = dataset[index]
        image = image.unsqueeze(0)
        if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
            image = F.interpolate(image,
                                   size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                   align_corners=False)
        with torch.no_grad():
            logits = target_model(image.cuda())
        if logits.max(1)[1].item() != label.item():
            raise Exception("error!")
            # index = np.random.randint(0, len(dataset))
            # image, true_label = dataset[index]
            # image = image.unsqueeze(0)
            # if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
            #     image = F.interpolate(image,
            #                        size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
            #                        align_corners=False)
            # with torch.no_grad():
            #     logits = target_model(image.cuda())
        assert true_label == label.item()
        images.append(torch.squeeze(image))
    return torch.stack(images) # B,C,H,W

def get_adv_images(images, true_labels, targeted, target_labels, dataset, model, attack_name, norm, max_query):
    target_images = get_image_of_target_class(dataset, target_labels, model)
    result = {}
    epsilon = 1.0
    if dataset.startswith("CIFAR"):
        epsilon = 1.0
    elif dataset == "TinyImageNet":
        epsilon = 2.0
    elif dataset == "ImageNet":
        epsilon = 10.0
    if norm == "linf":
        epsilon = 8/255
    for index, query in enumerate(np.arange(1000,max_query+1, 1000)):
        attacker = construct_attack(query, model, dataset, target_labels, attack_name,targeted, norm, epsilon)
        if attack_name == "HSJA" or ("Tangent" in attack_name and "QEBA" not in attack_name):
            adv_images, _, _, _, _ = attacker.attack(index, images, target_images, true_labels, target_labels)
        elif attack_name == "BA":
            image = images.numpy()[0]
            label = true_labels.item()
            ba_result = attacker(input_or_adv=image,
                            label=label,
                            unpack=False,
                            iterations=1200,
                            max_directions=25,
                            max_queries=query,
                            starting_point=target_images.detach().cpu().numpy()[0],
                            initialization_attack=None,  # foolbox default
                            log_every_n_steps=100,
                            spherical_step=0.01,
                            source_step=0.01,
                            step_adaptation=1.5,
                            batch_size=1,
                            tune_batch_size=True,
                            threaded_rnd=True,
                            threaded_gen=True,
                            alternative_generator=False,  # foolbox default
                            internal_dtype=eval('np.float32'),
                            save_all_steps=False,
                            verbose=False)
            adv_images = torch.from_numpy(ba_result.perturbed)
        elif attack_name == "SignOPT" or attack_name=="SVMOPT":
            if targeted:
                adv_images, _, _,_, _, _ = attacker.targeted_attack(
                    index, images.cuda(), target_labels)
            else:
                adv_images, _, _, _, _, _ = attacker.untargeted_attack(
                        index,  images.cuda(), true_labels)
            adv_images = adv_images.squeeze(0)

        elif attack_name.startswith("QEBA"):
            if targeted:
                attacker._default_criterion = TargetClass(target_labels[0].item())
                a = Adversarial(model, attacker._default_criterion, images, true_labels[0].item(),
                                distance=attacker._default_distance, threshold=attacker._default_threshold,
                                targeted_attack=targeted)
            else:
                target_labels = None
                attacker._default_criterion = Misclassification()
                a = Adversarial(model, attacker._default_criterion, images, true_labels[0].item(),
                                distance=attacker._default_distance, threshold=attacker._default_threshold,
                                targeted_attack=targeted)
                attacker.external_dtype = a.unperturbed.dtype

                def decision_function(x):
                    out = a.forward(x, strict=False)[1]  # forward function returns pr
                    return out

                target_images, num_calls = attacker.initialize(model, images.squeeze(0), decision_function, None,
                                                           true_labels, target_labels)
            attacker._starting_point = target_images[0]

            adv_images, _, _, _, _ = attacker.attack(index, a)

        adv_image = np.transpose(adv_images.detach().cpu().numpy(),(1,2,0))  # C,H,W --> H,W,C
        adv_image = np.uint8(np.round(np.clip(adv_image * 255., 0, 255)))
        result[query] = adv_image
    return result

def pil_loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    image_path = "/home1/machen/dataset/ILSVRC2017/Data/CLS-LOC/train/n02051845/n02051845_9729.JPEG"
    true_label = torch.LongTensor([439])
    target_class_label = torch.LongTensor([153])  # Maltese dog
    images = pil_loader(image_path)
    transform = transforms.Compose([transforms.Resize(size=(224, 224)),transforms.ToTensor()])
    images = transform(images)
    images = images.unsqueeze(0)
    model = StandardModel(args.dataset, "resnet101", no_grad=True).cuda()
    model = model.eval()
    result = get_adv_images(images, true_label,args.targeted, target_class_label, args.dataset, model, args.attack,
                            args.norm, args.max_queries)
    folder = "/home1/machen/hard_label_attacks/paper_chinese_figures/demo_adv_images/"
    os.makedirs(folder, exist_ok=True)
    for query, adv_image in result.items():
        file_path = "{}/{}_{}.png".format(folder, args.attack, query)
        PIL.Image.fromarray(adv_image).convert('RGB').save(file_path,"png")