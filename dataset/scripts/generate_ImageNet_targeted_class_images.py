import argparse
import glob
import os
import sys
from collections import defaultdict

sys.path.append(os.getcwd())
import torch
import numpy as np
import glog as log
from config import PY_ROOT, MODELS_TRAIN_STANDARD, MODELS_TEST_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from models.standard_model import StandardModel, MetaLearnerModelBuilder
from torch.nn import functional as F

def generate_attacked_dataset(dataset, num_sample_per_class, models):
    selected_images = []
    selected_true_labels = []
    selected_img_id = []
    total_count = defaultdict(int)
    data_loader = DataLoaderMaker.get_imgid_img_label_data_loader(dataset, 10, is_train=False, shuffle=False)
    log.info("begin select")
    if dataset != "ImageNet":
        for idx, (image_id, images, labels) in enumerate(data_loader):
            log.info("read {}-th batch images".format(idx))
            images_gpu = images.cuda()
            pred_eq_true_label = []
            continue_this_class = False
            for label in labels:
                if total_count[label.item()] > num_sample_per_class:
                    continue_this_class = True
                    break
            if continue_this_class:
                continue
            for model in models:
                model.cuda()
                with torch.no_grad():
                    logits = model(images_gpu)
                model.cpu()
                pred = logits.max(1)[1]
                correct = pred.detach().cpu().eq(labels).long()
                pred_eq_true_label.append(correct.detach().cpu().numpy())
            pred_eq_true_label = np.stack(pred_eq_true_label).astype(np.uint8) # M, B
            pred_eq_true_label = np.bitwise_and.reduce(pred_eq_true_label, axis=0)  # 1,0,1,1,1
            for index in np.where(pred_eq_true_label)[0]:
                total_count[labels[index].item()] += 1
            selected_image = images.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]]
            selected_images.append(selected_image)
            selected_true_labels.append(labels.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]])
            selected_img_id.append(image_id.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]])
    else:
        for idx, (image_id, images, labels) in enumerate(data_loader):
            log.info("read {}-th batch images".format(idx))
            pred_eq_true_label = []
            continue_this_class = False
            if len(total_count) == 1000:
                all_full = True
                for key, count in total_count.items():
                    if count < num_sample_per_class:
                        all_full = False
                if all_full:
                    break
            for label in labels:
                if total_count[label.item()] > num_sample_per_class:
                    continue_this_class = True
                    break

            if continue_this_class:
                continue
            for model in models:
                if model.input_size[-1] != 299:
                    images_gpu = F.interpolate(images, size=(model.input_size[-2], model.input_size[-1]),
                                           mode='bilinear', align_corners=False)
                    images_gpu = images_gpu.cuda()  # 3 x 299 x 299
                else:
                    images_gpu = images.cuda()
                with torch.no_grad():
                    model.cuda()
                    logits = model(images_gpu)
                    model.cpu()
                pred = logits.max(1)[1]
                correct = pred.detach().cpu().eq(labels).long()
                pred_eq_true_label.append(correct.detach().cpu().numpy())
            pred_eq_true_label = np.stack(pred_eq_true_label).astype(np.uint8) # M, B
            pred_eq_true_label = np.bitwise_and.reduce(pred_eq_true_label, axis=0)  # 1,0,1,1,1
            for index in np.where(pred_eq_true_label)[0]:
                total_count[labels[index].item()] += 1
            selected_image = images.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]]
            selected_images.append(selected_image)
            selected_true_labels.append(labels.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]])
            selected_img_id.append(image_id.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]])

    selected_images = np.concatenate(selected_images, 0)
    selected_true_labels = np.concatenate(selected_true_labels, 0)
    selected_img_id = np.concatenate(selected_img_id, 0)

    selected_images = selected_images[:num_sample_per_class]
    selected_true_labels = selected_true_labels[:num_sample_per_class]
    selected_img_id = selected_img_id[:num_sample_per_class]
    return selected_images, selected_true_labels, selected_img_id

def save_selected_images(selected_images, selected_true_labels, selected_img_id, save_path):
    np.savez(save_path, images=selected_images, labels=selected_true_labels, image_id=selected_img_id)

def load_models(dataset):
    archs = []
    model_path_list = []

    if dataset == "CIFAR-10" or dataset == "CIFAR-100":
        for arch in MODELS_TEST_STANDARD[dataset]:
            test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(
                PY_ROOT, dataset, arch)
            if os.path.exists(test_model_path):
                archs.append(arch)
                model_path_list.append(test_model_path)
            else:
                log.info(test_model_path + " does not exist!")
    elif dataset == "TinyImageNet":
        # for arch in ["vgg11_bn","resnet18","vgg16_bn","resnext64_4","densenet121"]:
        for arch in list(set(MODELS_TEST_STANDARD[dataset] + MODELS_TRAIN_STANDARD[dataset])):
            test_model_path = "{}/train_pytorch_model/real_image_model/{}@{}@*.pth.tar".format(
                PY_ROOT, dataset, arch)
            test_model_path = list(glob.glob(test_model_path))[0]
            if os.path.exists(test_model_path):
                archs.append(arch)
                model_path_list.append(test_model_path)
            else:
                log.info(test_model_path + "does not exist!")
    else:
        log.info("begin check arch")
        for arch in list(set(MODELS_TEST_STANDARD[dataset] + MODELS_TRAIN_STANDARD[dataset])):
            test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}*.pth".format(
                PY_ROOT, dataset, arch)
            test_model_path = list(glob.glob(test_model_list_path))
            if len(test_model_path) == 0:  # this arch does not exists in args.dataset
                continue
            archs.append(arch)
            model_path_list.append(test_model_path[0])
            log.info("check arch {} done, archs length {}".format(arch, len(archs)))
    models = []
    log.info("begin construct model")
    if dataset == "TinyImageNet":
        for idx, arch in enumerate(archs):
            model = MetaLearnerModelBuilder.construct_tiny_imagenet_model(arch, dataset)
            model_path = model_path_list[idx]
            model.load_state_dict(torch.load(model_path, map_location=lambda storage, location: storage)["state_dict"])
            model.eval()
            models.append(model)
            log.info("load {} over".format(arch))
    else:
        for arch in archs:
            model = StandardModel(dataset, arch, no_grad=True)
            model.eval()
            models.append(model)
            log.info("load {} over".format(arch))
    log.info("end construct model")
    return models

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"])
    args = parser.parse_args()
    dataset = args.dataset
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    save_path = "{}/attacked_images/{}/{}_target_class_images.npz".format(PY_ROOT, dataset, dataset)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_log_file(os.path.dirname(save_path)+"/generate_{}_target_class_images.log".format(dataset))
    models = load_models(dataset)
    selected_images, selected_true_labels, selected_img_id = generate_attacked_dataset(dataset, 100, models)

    save_selected_images(selected_images, selected_true_labels, selected_img_id, save_path)
    print("done")
