# Introduction
This repository includes the source code for "Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks",
which is published in NeurIPS 2021.

[[Paper]](https://arxiv.org/abs/2111.07492) [[Slides]](https://raw.githubusercontent.com/machanic/TangentAttack/main/paper_materials/slides.pdf) [[Poster]](https://raw.githubusercontent.com/machanic/TangentAttack/main/paper_materials/poster.pdf)

# Citation
We kindly ask anybody who uses this code to cite the following bibtex:

```
@inproceedings{ma2021finding,
 author = {Ma, Chen and Guo, Xiangyu and Chen, Li and Yong, Jun-Hai and Wang, Yisen},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {19288--19300},
 publisher = {Curran Associates, Inc.},
 title = {Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks},
 url = {https://proceedings.neurips.cc/paper/2021/file/a113c1ecd3cace2237256f4c712f61b5-Paper.pdf},
 volume = {34},
 year = {2021}
}
```

# Structure of Folders and Files
```
+-- configures
|   |-- HSJA.json  # the hyperparameters setting of HSJA, which is also used in Tangent Attack and Generalized Tangent Attack.
+-- dataset
|   |-- dataset_loader_maker.py  # it returns the data loader class that includes 1000 attacks images for the experiments.
|   |-- npz_dataset.py  # it is the dataset class that includes 1000 attacks images for the experiments.
+-- models
|   |-- defensive_model.py # the wrapper of defensive networks (e.g., AT, ComDefend, Feature Scatter), and it converts the input image's pixels to the range of 0 to 1 before feeding.
|   |-- standard_model.py # the wrapper of standard classification networks, and it converts the input image's pixels to the range of 0 to 1 before feeding.
+-- tangent_attack_hemisphere # the folder of Tangent Attack (TA) 
|   |-- attack.py  # the main class for the attack.
|   |-- tangent_point_analytical_solution.py  # the class for computing the optimal tagent point of the hemisphere.
+-- tangent_attack_semiellipsoid  # the folder of Generailized Tangent Attack (G-TA)
|   |-- attack.py  # the main class for the attack.
|   |-- tangent_point_analytical_solution.py  # the class for computing the optimal tagent point of the semi-ellipsoid.
+-- cifar_models   # this folder includes the target models of CIFAR-10, i.e., PyramidNet-272, GDAS, WRN-28, and WRN-40 networks.
|-- config.py   # the main configuration of Tangent Attack.
|-- logs  # all the output (logs and result stats files) are located inside this folder
|-- train_pytorch_model  # the pretrained weights of target models
|-- attacked_images  # the 1000 image data for evaluation 
```
The folder of `attacked_images` contains the 1000 tested images, which are packaged into `.npz` format with the pixel range of `[0-1]`.
This folder can be downloaded from [https://drive.google.com/file/d/1NkfMPShB9dUuugyFr2T8KTKM4kdwfKC2/view?usp=sharing](https://drive.google.com/file/d/1NkfMPShB9dUuugyFr2T8KTKM4kdwfKC2/view?usp=sharing).

The folder of `train_pytorch_model` contains the pretrained weights of target models, which can be downloaded from [https://drive.google.com/file/d/1VfCdU14nAhOvumXTIA-B8OC6XwGYvUml/view?usp=sharing](https://drive.google.com/file/d/1VfCdU14nAhOvumXTIA-B8OC6XwGYvUml/view?usp=sharing).

In the attack, all logs are dumped to `logs` folder. The results of attacks are also written into the `logs` folder, which use the `.json` format.
I have uploaded the compressed zip file of the experimental results onto [https://drive.google.com/file/d/1JswjvdDpaWMU7keGLA5HaVnZkYP3LpUO/view?usp=sharing](https://drive.google.com/file/d/1JswjvdDpaWMU7keGLA5HaVnZkYP3LpUO/view?usp=sharing),
so that you can directly use the results of baseline methods without repeatedly running experiments.

# Attack Command

The following command could run Tangent Attack (TA) and Generalized Tangent Attack (G-TA) on the CIFAR-10 dataset under the untargetd attack's setting:

```
python tangent_attack_hemisphere/attack.py --gpu 0 --norm l2 --dataset CIFAR-10 --arch resnet-50
python tangent_attack_hemisphere/attack.py --gpu 0 --norm l2 --dataset CIFAR-10 --arch gdas
python tangent_attack_semiellipsoid/attack.py --gpu 0 --norm l2 --dataset CIFAR-10 --arch resnet-50
python tangent_attack_semiellipsoid/attack.py --gpu 0 --norm l2 --dataset CIFAR-10 --arch gdas
```
Once the attack is running, it directly writes the `log` into a newly created `logs` folder. After attacking, the statistical result are also dumped into the same folder, which is named as `*.json` file. 

Also, you can use the following bash shell to run the attack of different models one by one.
```
./tangent_attack_CIFAR_undefended_models.sh
```
The commmand of attacks of defense models are presented in `tangent_attack_CIFAR_defense_models.sh`.

* The gpu device could be specified by the ```--gpu device_id``` argument.
* the targeted attack can be specified by the `--targeted` argument. If you want to perform untargeted attack, just don't pass it.
* the attack of defense models uses `--attack_defense --defense_model adv_train/jpeg/com_defend/TRADES` argument.
# Requirement
Our code is tested on the following environment (probably also works on other environments without many changes):

* Ubuntu 18.04
* Python 3.7.3
* CUDA 11.1
* CUDNN 8.0.4
* PyTorch 1.7.1
* torchvision 0.8.2
* numpy 1.18.0
* pretrainedmodels 0.7.4
* bidict 0.18.0
* advertorch 0.1.5
* glog 0.3.1

You can just type `pip install -r requirements.txt` to install packages.

# Download Files of Pre-trained Models and Running Results
In summary, there are three extra folders that can be downloaded, i.e., `attacked_images`, `train_pytorch_model`, and optionally `logs`.

The `attacked_images` can be downloaded from [https://drive.google.com/file/d/1NkfMPShB9dUuugyFr2T8KTKM4kdwfKC2/view?usp=sharing](https://drive.google.com/file/d/1NkfMPShB9dUuugyFr2T8KTKM4kdwfKC2/view?usp=sharing).
Besides, the targeted attack requires a randomly selected image of the target class in the validation set, and the validation set of the ImageNet dataset can be downloaded from [https://drive.google.com/file/d/1sE1i25mXApKuBChdhSqLbcFUEXTYYF9H/view?usp=sharing](https://drive.google.com/file/d/1sE1i25mXApKuBChdhSqLbcFUEXTYYF9H/view?usp=sharing).

The pre-trained weights of the target models can be downloaded from [https://drive.google.com/file/d/1VfCdU14nAhOvumXTIA-B8OC6XwGYvUml/view?usp=sharing](https://drive.google.com/file/d/1VfCdU14nAhOvumXTIA-B8OC6XwGYvUml/view?usp=sharing).

Before running experiments, please download above files and uncompress them to this project's root path.

Also, I have uploaded the compressed zip format file that contains running logs and experimental results onto [https://drive.google.com/file/d/1JswjvdDpaWMU7keGLA5HaVnZkYP3LpUO/view?usp=sharing](https://drive.google.com/file/d/1JswjvdDpaWMU7keGLA5HaVnZkYP3LpUO/view?usp=sharing).

You can use these experimental results directly without spending a lot of time to re-run the experiments.
