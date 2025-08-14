"""
Modified Version of Supervised Contrastive Learning
https://github.com/HobbitLong/SupContrast

@Article{khosla2020supervised,
    title   = {Supervised Contrastive Learning},
    author  = {Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and Dilip Krishnan},
    journal = {arXiv preprint arXiv:2004.11362},
    year    = {2020},
}
"""

import os
import sys
import argparse
import time

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.cuda.amp import GradScaler, autocast

from utils import (
    Transform_mix,
    AverageMeter,
)
import numpy as np
from utils import adjust_learning_rate
from utils import set_optimizer, save_model, select_images, get_rand_idx
from torch.utils.data.sampler import SubsetRandomSampler
from networks.resnet_big import SupConResNet
from losses import SupConLoss_robust

from rep_attack import SupConRepAdv


def parse_option():
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=512, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=4, help="num of workers to use")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")

    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--lr_decay_epochs",
        type=str,
        default="100,150",
        help="where to decay lr, can be a list",
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")

    # model dataset
    parser.add_argument("--model", type=str, default="resnet18")

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "path"],
        help="dataset",
    )

    parser.add_argument("--trans_order", type=str, default="b0, s0, t0", help="transform order")
    parser.add_argument("--atk_anchor", type=str, default="t0, s0", help="atk anchor set")
    parser.add_argument(
        "--atk_randstart", type=str, default="", help="atk anchor set with random perturbation"
    )
    parser.add_argument("--atk_contrast", type=str, default="t0, s0, b0", help="atk contrast set")
    parser.add_argument("--cln_anchor", type=str, default="b0", help="atk contrast mode")

    parser.add_argument("--atk_eps", type=float, default=8, help="atk epsilon size")
    parser.add_argument("--atk_alpha", type=float, default=2, help="atk step size")
    parser.add_argument("--attack_steps", type=int, default=10, help="inner maximization step num")
    parser.add_argument(
        "--atk_type", type=str, default="Linf", choices=["Linf", "L2"], help="choose method"
    )

    parser.add_argument("--atk_sc", action="store_true", help="atk_self_contrast")
    parser.add_argument("--data_folder", type=str, default="./data", help="path to custom dataset")

    # temperature
    parser.add_argument("--temp", type=float, default=0.1, help="temperature for loss function")

    # other setting
    parser.add_argument("--cosine", action="store_true", help="using cosine annealing")
    parser.add_argument("--name", type=str, default="from_scratch", help="encoder name")

    opt = parser.parse_args()

    opt.atk_eps /= 255
    opt.atk_alpha /= 255
    print(opt.atk_eps)
    print(opt.atk_alpha)

    opt.trans_order = opt.trans_order.replace(" ", "").split(",")
    opt.atk_anchor = opt.atk_anchor.replace(" ", "").split(",")
    opt.atk_randstart = opt.atk_randstart.replace(" ", "").split(",")
    opt.atk_contrast = opt.atk_contrast.replace(" ", "").split(",")
    opt.cln_anchor = opt.cln_anchor.replace(" ", "").split(",")

    opt.model_path = "./ckpt/{}".format(opt.dataset)
    opt.tb_path = "./tensorboard/encoder/{}_tensorboard".format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.tb_folder = os.path.join(opt.tb_path, opt.name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader

    transform_base = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_sim = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
    )
    transform_triv = transforms.Compose(
        [
            transforms.TrivialAugmentWide(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    train_transform = Transform_mix(transform_base, transform_sim, transform_triv)
    train_transform.parse_order(opt.trans_order)

    if opt.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=opt.data_folder, transform=train_transform, download=True
        )
    elif opt.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=opt.data_folder, transform=train_transform, download=True
        )
    elif opt.dataset == "path":
        train_dataset = datasets.ImageFolder(root=opt.data_folder, transform=train_transform)
    else:
        raise ValueError(opt.dataset)

    if opt.dataset == "cifar10":
        train_indices = list(range(50000))
        val_indices = []
        count = np.zeros(10)
        for index in range(len(train_dataset)):
            _, target = train_dataset[index]
            if np.all(count == 100):
                break
            if count[target] < 100:
                count[target] += 1
                val_indices.append(index)
                train_indices.remove(index)

    elif opt.dataset == "cifar100":
        train_indices = list(range(50000))
        val_indices = []
        count = np.zeros(100)
        for index in range(len(train_dataset)):
            _, target = train_dataset[index]
            if np.all(count == 10):
                break
            if count[target] < 10:
                count[target] += 1
                val_indices.append(index)
                train_indices.remove(index)
    print("Overlap indices:", list(set(train_indices) & set(val_indices)))
    print("Size of train set:", len(train_indices))
    print("Size of val set:", len(val_indices))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        sampler=SubsetRandomSampler(train_indices),
        pin_memory=True,
        num_workers=opt.num_workers,
    )

    return train_loader


def set_model(opt):

    model = SupConResNet(name=opt.model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)

        model = model.cuda()
        cudnn.benchmark = True

    return model


def train_encoder_robust(train_loader, atk, model, optimizer, epoch, opt):
    """one epoch training

    use transformed images with different augmentations

    """
    torch.autograd.set_detect_anomaly(True)
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    adv_losses = AverageMeter()

    end = time.time()
    # scaler = GradScaler() #use for fast training

    criterion = SupConLoss_robust(temperature=opt.temp)

    for idx, (images, labels) in enumerate(train_loader):

        images_all = [img.cuda() for img in images]
        labels = labels.cuda()

        atk_anchor = select_images(images_all, opt.trans_order, opt.atk_anchor)
        atk_contrast = select_images(images_all, opt.trans_order, opt.atk_contrast)
        cln_anchor = select_images(images_all, opt.trans_order, opt.cln_anchor)

        rand_idx = get_rand_idx(opt.atk_anchor, opt.atk_randstart)

        adv_images = atk.perturb(atk_anchor, atk_contrast, labels, rand_idx=rand_idx)

        cln_cat = torch.cat(cln_anchor, dim=0)

        images = torch.cat([adv_images, cln_cat], dim=0)

        bsz = labels.shape[0]

        features = model(images)

        num_adv = len(atk_anchor) * bsz
        num_cln = len(cln_anchor) * bsz

        (adv_f, cln_f) = torch.split(features, [num_adv, num_cln], dim=0)

        adv_f_split = torch.split(adv_f, bsz, dim=0)
        cln_f_split = torch.split(cln_f, bsz, dim=0)

        adv_f_cat = torch.cat([f.unsqueeze(1) for f in adv_f_split], dim=1)
        cln_f_cat = torch.cat([f.unsqueeze(1) for f in cln_f_split], dim=1)

        anchor_adv = adv_f_cat
        contrast_adv = torch.cat([adv_f_cat, cln_f_cat], dim=1)

        anchor_cln = cln_f_cat
        contrast_cln = adv_f_cat

        clean_loss = criterion(anchor_cln, contrast_cln, labels, sc=True)
        adv_loss = criterion(anchor_adv, contrast_adv, labels)

        loss = clean_loss + adv_loss

        # update metric
        losses.update(loss.item(), bsz)
        adv_losses.update(adv_loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## use for fast training
        # scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # scaler.step(optimizer)
        # scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print(
                "Robust Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "adv_loss {adv_loss.val:.3f} ({adv_loss.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    adv_loss=adv_losses,
                )
            )
            sys.stdout.flush()

    return losses.avg, adv_losses.avg


def main():

    opt = parse_option()

    print(opt)
    # build data loader
    train_loader = set_loader(opt)

    model = set_model(opt)
    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    start_time = time.time()

    atk = SupConRepAdv(
        model,
        epsilon=opt.atk_eps,
        atk_temp=opt.temp,
        steps=opt.attack_steps,
        alpha=opt.atk_alpha,
    )

    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()

        loss, adv_train_loss = train_encoder_robust(train_loader, atk, model, optimizer, epoch, opt)

        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value("loss", loss, epoch)
        logger.log_value("adv_train_loss", adv_train_loss, epoch)
        logger.log_value("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        logger.log_value("epoch_time", time2 - time1, epoch)

        if epoch % opt.save_freq == 0:
            save_loss = str(round(loss, 3)).replace(".", "-")
            save_file = os.path.join(
                opt.save_folder,
                "ckpt_epoch_{epoch}_loss_{loss}.pth".format(epoch=epoch, loss=save_loss),
            )

            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model

    save_file = os.path.join(opt.save_folder, "last.pth")
    save_model(model, optimizer, opt, opt.epochs, save_file)

    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)

if __name__ == "__main__":
    main()
