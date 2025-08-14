import os
import argparse
import csv

import torch
from torchvision import transforms, datasets

import pandas as pd

from utils_eval import CIFAR10_ood, CIFAR100_ood
from pytorch_ood.detector import MaxSoftmax

from utils import AverageMeter
from networks.resnet_big import LinearClassifier , SupConResNet_eval

from torchattacks import PGD
from robustbench.utils import clean_accuracy
from robustbench.data import load_cifar10c, load_cifar100c
from robustbench.eval import benchmark
import numpy as np


def test_model(model, testloader):

    device = "cuda"
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print("Test accuracy: %.2f %%" % (100 * acc))
    return acc

def set_test_loader(opt):

    if opt.dataset == "cifar10":
        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        val_dataset = datasets.CIFAR10(root="./data/", train=False, transform=val_transform)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    else:
        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        val_dataset = datasets.CIFAR100(root="./data/", train=False, transform=val_transform)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )

    return val_loader


def test_corruption_robust(opt, model):

    # corruption type -> 3 noise, 4 blur, 4 weather, 4 digital

    model = model.cuda()
    model.eval()

    csv_name = "./robustness/{}/{}_corrupt_acc.csv".format(opt.dataset,opt.log_name)
    head_list = ["loss_type", "model", "linear", "name", "clean"]

    corrupt_acc = dict()
    corrupt_acc["loss_type"] = opt.loss_type
    corrupt_acc["model"] = os.path.basename(os.path.dirname(opt.model_ckpt))
    corrupt_acc["linear"] = os.path.basename(os.path.dirname(opt.linear_ckpt))
    corrupt_acc["name"] = opt.log_name

    test_loader = set_test_loader(opt)
    clean_acc = test_model(model, test_loader)
    corrupt_acc["clean"] = clean_acc

    corruptions_list = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]
    model = model.cuda()

    for corruption in corruptions_list:
        for severity in [1, 2, 3, 4, 5]:

            c_name = corruption + "_" + severity.__str__()
            head_list.append(c_name)
            if opt.dataset == "cifar10":
                x_test, y_test = load_cifar10c(
                    n_examples=10000, corruptions=[corruption], severity=severity,data_dir='./data/'
                )
            else:
                x_test, y_test = load_cifar100c(
                    n_examples=10000, corruptions=[corruption], severity=severity, data_dir='./data/'
                )
            x_test = x_test.cuda()

            acc = clean_accuracy(model.cuda(), x_test.cuda(), y_test.cuda())


            print(f"CIFAR-10 corruption: {corruption} severity: {severity} accuracy: {acc:.1%}")

            corrupt_acc[f"{c_name}"] = acc


    avg = np.mean(list(corrupt_acc.values())[5:])
    corrupt_acc["avg"] = avg
    head_list.append("avg")

    print("save corruption acc csv")
    with open(csv_name, "a", newline="") as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=head_list)

        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow(corrupt_acc)

    return avg




def test_autoattack_all(opt, model):

    model.eval()
    log_name = "./robustness/{}/{}.txt".format(opt.dataset,opt.log_name)
    device = torch.device("cuda:0")

    clean_acc, robust_acc = benchmark(
        model,
        dataset=opt.dataset,
        threat_model="Linf",
        batch_size=512,
        eps=8 / 255,
        log_path=log_name,
        device=device,
    )
  

    print(robust_acc)
    return robust_acc



def test_pgd_all(opt, model):

    
    test_loader = set_test_loader(opt)

    model.eval()

    device = "cuda"
    top1 = AverageMeter()

    for idx, (images, labels) in enumerate(test_loader):
        atk = PGD(model, eps=8 / 255, alpha=2 / 255, steps=20, random_start=True)
        adv_images = atk(images.to(device), labels.to(device))
        acc = clean_accuracy(model, adv_images.to(device), labels.to(device))
        top1.update(acc, 1)

        print("Linf 8/255 Robust accuracy:{} : {:.1%}".format(idx, acc))

    print("Linf 8/255 Robust accuracy for all set : {:.1%}".format(top1.avg))

    return top1.avg



def test_ood_all(opt, model_classifier):

    device = "cuda:0"
    loader_kwargs = {"batch_size": 256, "num_workers": 4}
    trans = transforms.Compose(
        [
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            
        ]
    )

    detectors = {
        "MSP": MaxSoftmax(model_classifier),
    }
    results = []
    if opt.dataset == "cifar10":
        benchmark = CIFAR10_ood(root="data", transform=trans)
        
    else:
        benchmark = CIFAR100_ood(root="data", transform=trans)
        
    with torch.no_grad():
        for detector_name, detector in detectors.items():
            print(f"> Evaluating {detector_name}")
            res = benchmark.evaluate(detector, loader_kwargs=loader_kwargs, device=device)
            for r in res:
                r.update({"Detector": detector_name})

            results += res
            

    df = pd.DataFrame(results)
    print((df.set_index(["Dataset", "Detector"]) * 100).to_csv(float_format="%.2f"))
    csv_file = "./robustness/{}/{}_ood.csv".format(opt.dataset,opt.log_name)
    df.to_csv(csv_file, index=False)

    auroc_avg = df["AUROC"].mean()
    print("mAUORC : ", auroc_avg)

    return auroc_avg


def test_robust(opt, model, test_loader):

    

    clean_acc = test_model(model, test_loader)

    csv_name = "./robustness/{}/robustness_all_{}.csv".format(opt.dataset,opt.log_name)
    head_list = [
        "model",
        "linear",
        "name",
        "SA",
        "AA",
        "PGD20",
        "mCA",
        "mAUROC"
    ]
    rob = dict()
    rob["model"] = os.path.basename(os.path.dirname(opt.model_ckpt))
    rob["linear"] = os.path.basename(os.path.dirname(opt.linear_ckpt))
    rob["name"] = opt.log_name
    rob["SA"] = clean_acc

    if opt.test_option == 'all':
        rob["AA"] = test_autoattack_all(opt, model)
        rob["PGD20"] = test_pgd_all(opt, model)
        rob["mCA"] = test_corruption_robust(opt, model)
        rob["mAUROC"] = test_ood_all(opt, model )
    elif opt.test_option == 'partial':
        if 'adversarial' in opt.test_type:
            rob["AA"] = test_autoattack_all(opt, model)
            rob["PGD20"] = test_pgd_all(opt, model)
        if 'corrupt' in opt.test_type:
            rob["mCA"] = test_corruption_robust(opt, model)
        if 'ood' in opt.test_type:
            rob["mAUROC"] = test_ood_all(opt, model)

        
    with open(csv_name, "a", newline="") as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=head_list)

        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow(rob)


def set_model(opt):


    if opt.dataset == 'cifar10':
        num_cls = 10
    else:
        num_cls = 100
    if opt.loss_type == "SupCon":
        model = SupConResNet_eval(opt.model,num_classes=num_cls)
        ckpt = torch.load(opt.model_ckpt, map_location="cpu")
        state_dict = ckpt["model"]

        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict

        model.load_state_dict(state_dict,strict=False)

        print("model load")

        linear_ckpt = torch.load(opt.linear_ckpt, map_location="cpu")
        state_dict = linear_ckpt["model"]
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict

        classifier = LinearClassifier(opt.model, num_cls)
        classifier.load_state_dict(state_dict)

        print("classifier load")
        model.fc = classifier

    model = model.cuda()
    model.eval()

    return model


def main(opt):

    if not os.path.exists("./robustness/{}".format(opt.dataset)):
        os.makedirs("./robustness/{}".format(opt.dataset))

    model = set_model(opt)
    test_loader = set_test_loader(opt)
    test_robust(opt, model, test_loader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--loss_type", type=str, default="SupCon", help="Loss Type. [SupCon, CE]")
    parser.add_argument("--model", type=str, default="resnet18", help="model arc")
    parser.add_argument("--dataset", type=str, default="cifar10", help="trained dataset")

    parser.add_argument("--model_ckpt", type=str, default="", help="path to pre-trained model")
    parser.add_argument(
        "--linear_ckpt", type=str, default="", help="path linear ckpt"
    )
    parser.add_argument(
        "--test_option",
        type=str,
        default="all",
        choices=["all", "partial"],
        help="Choose 'all' to evaluate all types of robustness, or 'partial' to evaluate specific types.",
    )
    parser.add_argument("--test_type", type=str, default="adversarial, corrupt, ood", help="Specify the types of robustness to evaluate. Applicable when 'test_option' is 'partial'. Options: 'adversarial', 'corrupt', 'ood'.")
    
    parser.add_argument("--batch_size", type=int, default=1024, help="test batch size")

    parser.add_argument("--log_name", type=str, default="test", help="name for log")

    opt = parser.parse_args()
    opt.test_type = opt.test_type.replace(" ", "").split(',')
    main(opt)
