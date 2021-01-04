IMAGE_SIZE = {"CIFAR-10":(32,32), "CIFAR-100":(32,32), "ImageNet":(224,224), "MNIST":(28, 28), "FashionMNIST":(28,28), "SVHN":(32,32),
              "TinyImageNet": (64,64)}
IN_CHANNELS = {"MNIST":1, "FashionMNIST":1, "CIFAR-10":3, "ImageNet":3, "CIFAR-100":3, "SVHN":3, "TinyImageNet":3}
CLASS_NUM = {"MNIST":10,"FashionMNIST":10, "CIFAR-10":10, "CIFAR-100":100, "ImageNet":1000, "SVHN":10, "TinyImageNet":200}

HOME_ROOT = "/home/machen"

PY_ROOT = "{}/hard_label_attacks".format(HOME_ROOT)

IMAGE_DATA_ROOT = {"CIFAR-10":"{}/dataset/CIFAR-10".format(HOME_ROOT), "CIFAR-100":"{}/dataset/CIFAR-100".format(HOME_ROOT),
                   'cifar10':"{}/dataset/CIFAR-10".format(HOME_ROOT),
                   "MNIST":"{}/dataset/MNIST".format(HOME_ROOT),
                   "FashionMNIST":"{}/dataset/FashionMNIST".format(HOME_ROOT),
                   "SVHN":"{}/dataset/SVHN".format(HOME_ROOT),
                   "ImageNet": "{}/dataset/ILSVRC2012/".format(HOME_ROOT),
                   "TinyImageNet": "{}/dataset/tinyImageNet/".format(HOME_ROOT)}

pretrained_cifar_model_conf = {"CIFAR-10":{
                                "vgg11_bn":None,
                                "vgg13_bn":None,
                                "vgg16_bn":None,
                                "vgg19_bn":None,
                                "alexnet":None,
                                "alexnet_bn":None,
                                "resnet-20":{"depth":20, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-32":{"depth":32, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-44":{"depth":44, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-50":{"depth":50, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-56":{"depth":56, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                               "resnet-110":{"depth":110, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                               "resnet-1202":{"depth":1202,"epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                               "preresnet-110":{"depth":110,"epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4,  "block_name":"BasicBlock"},
                                "resnext-8x64d":{"depth":29, "cardinality":8, "widen_factor":4, "schedule":[150,225],"wd":5e-4,"gamma":0.1, "drop":0},
                                "resnext-16x64d":{"depth":29, "cardinality":16, "widen_factor":4, "schedule":[150,225],"wd":5e-4,"gamma":0.1,  "drop":0},
                                "WRN-28-10":{"depth":28, "widen_factor":10, "drop":0.0,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-28-10-drop":{"depth":28, "widen_factor":10, "drop":0.3,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-34-10":{"depth":34, "widen_factor":10, "drop":0.0,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-34-10-drop":{"depth":34, "widen_factor":10, "drop":0.3,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-40-10-drop":{"depth":40, "widen_factor":10, "drop":0.3,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-40-10":{"depth":40, "widen_factor":10, "drop":0.0,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "densenet-bc-100-12":{"depth":100,"growthRate":12,"train_batch":64,"epochs":300, "schedule":[150,225],"wd":1e-4,"gamma":0.1,"compressionRate":2,"drop":0},
                                "densenet-bc-L190-k40":{"depth":190,"growthRate":40,"train_batch":64,"epochs":300, "schedule":[150,225],"wd":1e-4,"gamma":0.1,"compressionRate":2,"drop":0},
                                 "pcl_resnet-110": {"depth":110},
                                "pcl_resnet-50": {"depth":50}
                            },
                            "CIFAR-100":{
                                "vgg11_bn":None,
                                "vgg13_bn":None,
                                "vgg16_bn":None,
                                "vgg19_bn":None,
                                "alexnet":None,
                                "alexnet_bn":None,
                                "resnet-20":{"depth":20, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-32":{"depth":32, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-44":{"depth":44, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-50":{"depth":50, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-56":{"depth":56, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                                "resnet-110":{"depth":110, "epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                               "resnet-1202":{"depth":1202,"epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4, "block_name":"BasicBlock"},
                               "preresnet-110":{"depth":110,"epochs":164, "schedule":[81,122], "gamma":0.1, "wd":1e-4,  "block_name":"BasicBlock"},
                                "resnext-8x64d":{"depth":29, "cardinality":8, "widen_factor":4, "schedule":[150,225],"wd":5e-4,"gamma":0.1, "drop":0},
                                "resnext-16x64d":{"depth":29, "cardinality":16, "widen_factor":4, "schedule":[150,225],"wd":5e-4,"gamma":0.1, "drop":0},
                                "WRN-28-10":{"depth":28, "widen_factor":10, "drop":0.0,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-28-10-drop":{"depth":28, "widen_factor":10, "drop":0.3,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-34-10":{"depth":34, "widen_factor":10, "drop":0.0,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-34-10-drop":{"depth":34, "widen_factor":10, "drop":0.3,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-40-10-drop":{"depth":40, "widen_factor":10, "drop":0.3,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "WRN-40-10":{"depth":40, "widen_factor":10, "drop":0.0,"epochs":200, "schedule":[60,120,160], "wd":5e-4, "gamma":0.2},
                                "densenet-bc-100-12":{"depth":100,"growthRate":12,"train_batch":64,"epochs":300, "schedule":[150,225],"wd":1e-4,"gamma":0.1,"compressionRate":2, "drop":0},
                                "densenet-bc-L190-k40":{"depth":190,"growthRate":40,"train_batch":64,"epochs":300, "schedule":[150,225],"wd":1e-4,"gamma":0.1,"compressionRate":2, "drop":0},
                                "pcl_resnet-110": {"depth": 110},
                                "pcl_resnet-50": {"depth": 50}
                            },
                            "TinyImageNet":{
                                "pcl_resnet110": {"depth":110},
                                "pcl_resnet50": {"depth":50}
                            }
            }



IMAGENET_ALL_MODELS = ["inception_v3","pnasnet5large","senet154","inceptionv4","xception","resnet101"]

MODELS_TRAIN_STANDARD = {"CIFAR-10": ["alexnet", "densenet-bc-100-12", "densenet-bc-L190-k40",  "preresnet-110",
                                      "resnext-16x64d","resnext-8x64d","vgg19_bn","resnet-20","resnet-32","resnet-44","resnet-50",
                                          "resnet-56","resnet-110","resnet-1202"],
                         "CIFAR-100": ["alexnet", "densenet-bc-100-12", "densenet-bc-L190-k40",  "preresnet-110",
                                      "resnext-16x64d","resnext-8x64d","vgg19_bn","resnet-20","resnet-32","resnet-44","resnet-50",
                                          "resnet-56","resnet-110","resnet-1202"],
                         "ImageNet": ["alexnet", "bninception","densenet121", "densenet161","densenet169", "densenet201","dpn68",
                                     "resnext101_32x4d","resnext101_64x4d","se_resnext101_32x4d",
                                      "se_resnext50_32x4d","squeezenet1_0","squeezenet1_1","vgg11","vgg11_bn","vgg13_bn","vgg13",
                                      "vgg16","vgg16_bn","vgg19_bn","vgg19"],
                         "TinyImageNet": ["vgg13","densenet169","vgg11_bn","resnet34","vgg19","vgg13_bn","vgg11","resnet18","vgg16",
                                          "vgg19_bn","densenet201","resnet101","densenet161","resnet50","vgg16_bn","resnet152"]}

MODELS_TEST_STANDARD = {"CIFAR-10": ["pyramidnet272", "gdas","WRN-28-10-drop","WRN-40-10-drop","densenet-bc-L190-k40"],
                        "CIFAR-100":["pyramidnet272", "gdas","WRN-28-10-drop","WRN-40-10-drop","densenet-bc-L190-k40"],
                        "ImageNet": ["resnext101_64x4d","inceptionv4","senet154","resnet101","inceptionv3","pnasnet5large"],
                        "TinyImageNet":["resnext64_4","densenet121","resnext32_4"]}



MODELS_TRAIN_WITHOUT_RESNET = {"CIFAR-10": ["alexnet", "densenet-bc-100-12", "densenet-bc-L190-k40",  "preresnet-110",
                                      "resnext-16x64d","resnext-8x64d","vgg19_bn"],
                         "CIFAR-100": ["alexnet", "densenet-bc-100-12", "densenet-bc-L190-k40",  "preresnet-110",
                                      "resnext-16x64d","resnext-8x64d","vgg19_bn"],
                         "ImageNet": ["alexnet", "bninception","densenet121", "densenet161","densenet169", "densenet201","dpn68",
                                     "resnext101_32x4d","resnext101_64x4d","se_resnext101_32x4d",
                                      "se_resnext50_32x4d","squeezenet1_0","squeezenet1_1","vgg11","vgg11_bn","vgg13_bn","vgg13",
                                      "vgg16","vgg16_bn","vgg19_bn","vgg19"],
                         "TinyImageNet": ["vgg13","densenet169","vgg11_bn","vgg19","vgg13_bn","vgg11","vgg16",
                                          "vgg19_bn","densenet201","densenet161","vgg16_bn"]}

