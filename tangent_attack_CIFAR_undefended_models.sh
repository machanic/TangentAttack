#!/usr/bin/env bash

# for untargeted attacks, use the following commands
python tangent_attack_hemisphere/attack.py --gpu 0 --norm l2 --dataset CIFAR-10 --all_archs
python tangent_attack_semiellipsoid/attack.py --gpu 0 --norm l2 --dataset CIFAR-10 --all_archs

# for untargeted attacks, use the following commands
python tangent_attack_hemisphere/attack.py --gpu 0 --norm l2 --dataset CIFAR-10 --all_archs --targeted
python tangent_attack_semiellipsoid/attack.py --gpu 0 --norm l2 --dataset CIFAR-10 --all_archs --targeted