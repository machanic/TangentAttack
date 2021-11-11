# Introduction
This repository includes the source code for "Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks", which is published in NeurIPS 2021.

# Citation
We kindly ask anybody who uses this code to cite the following bibtex:

```
@inproceedings{
    ma2021finding,
    title={Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks},
    author={Chen Ma and Xiangyu Guo and Li Chen and Jun-Hai Yong and Yisen Wang},
    booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
    year={2021},
    url={https://openreview.net/forum?id=g0wang64Zjd}
}
```

# Structure of Folders and Files
```
+-- configures
|   |-- HSJA.json  # the hyperparameters setting of HSJA, which is also used in Tangent Attack
+-- dataset
|   |-- dataset_loader_maker.py  # it returns the data loader class that includes 1000 attacks images for the experiments.
|   |-- npz_dataset.py  # it is the dataset class that includes 1000 attacks images for the experiments.
+-- models
|   |-- defensive_model.py # the wrapper of defensive networks (e.g., AT, ComDefend, Feature Scatter), and it converts the input image's pixels to the range of 0 to 1 before feeding.
|   |-- standard_model.py # the wrapper of standard classification networks, and it converts the input image's pixels to the range of 0 to 1 before feeding.
+-- tangent_attack_hemisphere
|   |-- attack.py  # the main class for the attack.
|   |-- tangent_point_analytical_solution.py  # the class for computing the optimal tagent point of the hemisphere.
+-- tangent_attack_semiellipsoid
|   |-- attack.py  # the main class for the attack.
|   |-- tangent_point_analytical_solution.py  # the class for computing the optimal tagent point of the semi-ellipsoid.
+-- cifar_models   # this folder includes the target models of CIFAR-10, i.e., PyramidNet-272, GDAS, WRN-28, and WRN-40 networks.
|-- config.py   # the main configuration of Tangent Attack.
|-- logs  # all the output (logs and result stats files) are located inside this folder
|-- train_pytorch_model  # the pretrained weights of target models
|-- attacked_images  # the 1000 image data for evaluation 
```
In general, the `train_pytorch_model` includes the pretrained models' weights, and `attacked_images` includes the image data, which is packaged into `.npz` format with pixel range of `[0-1]`.

In the attack, all logs are dumped to `logs` folder, the statistical results are also written into `logs` folder, which are `.json` format.

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
./tangent_attack_CIFAR_normal_models.sh
```
The commmand of attacks of defensive models are presented in `tangent_attack_CIFAR_defensive_models.sh`.


* The gpu device could be specified by the ```--gpu device_id``` argument.
* the targeted attack can be specified by the `--targeted` argument. If you want to perform untargeted attack, just don't pass it.
* the attack of defensive model uses `--attack_defense --defense_model adv_train/jpeg/com_defend/TRADES` argument.
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

# Download Files of Running Results and Logs
I have uploaded all the logs and results with the compressed zip file format onto [this google drive link](https://drive.google.com/file/d/1vng1Gs6YgZs3PGMvfJb-exRRIsrbo5vx/view?usp=sharing) so that you can download them.
