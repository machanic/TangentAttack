#!/usr/bin/env bash

# for untargeted attacks, use the following commands
python tangent_attack_hemisphere/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model adv_train --arch resnet-50
python tangent_attack_hemisphere/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model TRADES --arch resnet-50
python tangent_attack_hemisphere/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model com_defend --arch resnet-50
python tangent_attack_hemisphere/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model jpeg --arch resnet-50

python tangent_attack_semiellipsoid/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model adv_train --arch resnet-50
python tangent_attack_semiellipsoid/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model TRADES --arch resnet-50
python tangent_attack_semiellipsoid/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model com_defend --arch resnet-50
python tangent_attack_semiellipsoid/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model jpeg --arch resnet-50

# for targeted attacks, use the following commands
python tangent_attack_hemisphere/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model adv_train --arch resnet-50 --targeted
python tangent_attack_hemisphere/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model TRADES --arch resnet-50 --targeted
python tangent_attack_hemisphere/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model com_defend --arch resnet-50 --targeted
python tangent_attack_hemisphere/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model jpeg --arch resnet-50 --targeted

python tangent_attack_semiellipsoid/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model adv_train --arch resnet-50 --targeted
python tangent_attack_semiellipsoid/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model TRADES --arch resnet-50 --targeted
python tangent_attack_semiellipsoid/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model com_defend --arch resnet-50 --targeted
python tangent_attack_semiellipsoid/attack.py --gpu 0 --norm l2 --dataset CIFAR-10  --attack_defense --defense_model jpeg --arch resnet-50 --targeted

