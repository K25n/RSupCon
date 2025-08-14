import os
import argparse
import torch

from torchvision import transforms, datasets
from networks.resnet_big import LinearClassifier, SupConResNet_eval

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

import numpy as np
from utils_vis import ClassSpecificImageGeneration , max_act_img


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
        raise ValueError(opt.dataset)


    return val_loader

def print_save_tsne(actual, cluster, name):

    plt.figure(figsize=(8, 8))
    cifar = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    for i, label in zip(range(10), cifar):

        idx = np.where(actual == i)
        plt.scatter(cluster[idx, 0], cluster[idx, 1], marker=".", s=20, label=label)
    plt.legend(fontsize="xx-large", markerscale=3.5)
    plt.tight_layout()
    # plt.show()
    if not os.path.isdir('./vis/tsne'):
        os.makedirs('./vis/tsne')
    file_name = "./vis/tsne/{}.png".format(name)
    plt.savefig(file_name, bbox_inches="tight", dpi=300)


def gen_actimg(model,name,rgb_set=False,color="random"):
    for i in range(10):
        r_csig = ClassSpecificImageGeneration(model, i,name,rgb_set,color)
        r_csig.generate()


def vis_tsne_supcon(model, testloader):

    actual = []
    deep_features = []
    model.eval()
    model = model.cuda()
    device = "cuda"
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            features = model(images) 

            deep_features += features.cpu().numpy().tolist()
            actual += labels.cpu().numpy().tolist()

    tsne = TSNE(n_components=2, random_state=0) 
    cluster = np.array(tsne.fit_transform(np.array(deep_features)))
    actual = np.array(actual)

    return actual, cluster

def vis_func(opt, model):
    test_loader = set_test_loader(opt)

    
    actual_en, cluster_en = vis_tsne_supcon(model.encoder, test_loader)
    print_save_tsne(actual_en, cluster_en, opt.log_name + "_encoder")

    

def main(opt):

    test_loader = set_test_loader(opt)
    if not os.path.isdir('./vis'):
        os.makedirs('./vis')

    model = set_model(opt)
    vis_func(opt, model)
    gen_actimg(model,opt.log_name)
    max_act_img(test_loader,model,opt.log_name)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--loss_type", type=str, default="SupCon", help="Loss Type. [SupCon, CE]")
    parser.add_argument("--model", type=str, default="resnet18", help="model arc")
    parser.add_argument("--dataset", type=str, default="cifar10", help="trained dataset")

    parser.add_argument("--model_ckpt", type=str, default="", help="path to pre-trained model")
    parser.add_argument(
        "--linear_ckpt", type=str, default="", help="if train with SupCon, path linear ckpt"
    )

    parser.add_argument("--batch_size", type=int, default=512, help="test batch size")

    parser.add_argument("--log_name", type=str, default="test", help="name for log name")

    opt = parser.parse_args()
    main(opt)

