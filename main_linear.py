"""
Modified Version of linear training
https://github.com/HobbitLong/SupContrast

@Article{khosla2020supervised,
    title   = {Supervised Contrastive Learning},
    author  = {Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and Dilip Krishnan},
    journal = {arXiv preprint arXiv:2004.11362},
    year    = {2020},
}
"""


from __future__ import print_function

import sys
import argparse
import time

import os
import torch

import torch.backends.cudnn as cudnn

import tensorboard_logger as tb_logger
from torchvision import transforms, datasets
from utils import AverageMeter
from utils import accuracy
from utils import set_optimizer, save_model
from torchattacks import PGD, PGDL2
from trades import trades_loss, trades_adv
import torch.nn.functional as F

from networks.resnet_big import SupConResNet, LinearClassifier


def parse_option():
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("--batch_size", type=int, default=512, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=16, help="num of workers to use")
    parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")

    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--lr_decay_epochs", type=str, default="60,75,90", help="where to decay lr, can be a list"
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.2, help="decay rate for learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")

    # model dataset
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="dataset",
    )
    parser.add_argument(
        "--linear_loss", type=str, default="CE", help="Loss Type. [CE,Madry,TRADES]"
    )
    parser.add_argument("--aug_type", type=str, default="trivial", help="[base/sim/trivial]")

    # other setting
    parser.add_argument("--ckpt", type=str, default="", help="path to pre-trained model")
    parser.add_argument("--name", type=str, default="", help="name of linear")
    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = "./data/"

    opt.model_path = "./ckpt/{}/".format(opt.dataset)
    opt.tb_path = "./tensorboard/linear/{}_tensorboard".format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = "linear"

    opt.tb_folder = os.path.join(opt.tb_path, opt.name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == "cifar10":
        opt.n_cls = 10
    elif opt.dataset == "cifar100":
        opt.n_cls = 100
    elif opt.dataset == "stl10":
        opt.n_cls = 10
    else:
        raise ValueError("dataset not supported: {}".format(opt.dataset))

    return opt


def set_model(opt):

    model = SupConResNet(name=opt.model)

    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)


    ckpt = torch.load(opt.ckpt, map_location="cpu")
    state_dict = ckpt["model"]

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def set_loader(opt):

    if opt.aug_type == "base":
        train_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    elif opt.aug_type == "sim":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )
    elif opt.aug_type == "trivial":
        train_transform = transforms.Compose(
            [
                transforms.TrivialAugmentWide(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    if opt.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=opt.data_folder, transform=train_transform, download=True
        )
        val_dataset = datasets.CIFAR10(root=opt.data_folder, train=False, transform=val_transform)
    elif opt.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=opt.data_folder, transform=train_transform, download=True
        )
        val_dataset = datasets.CIFAR100(root=opt.data_folder, train=False, transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)

        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                )
            )
            sys.stdout.flush()

    return losses.avg, top1.avg


def train_trades_linear(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    # model.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model_classifier = torch.nn.Sequential(model.encoder, classifier)

    criterion_kl = torch.nn.KLDivLoss(reduction="sum")

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        adv_images = trades_adv(
            model=model_classifier,
            x_natural=images,
            y=labels,
            step_size=0.007,
            epsilon=0.031,
            perturb_steps=10,
            beta=6,
        )
        model.eval()
        classifier.train()

        # compute loss
        with torch.no_grad():
            clean_features = model.encoder(images)
            adv_features = model.encoder(adv_images)
        clean_output = classifier(clean_features.detach())
        adv_output = classifier(adv_features.detach())
        adv_loss = (1.0 / bsz) * criterion_kl(
            F.log_softmax(adv_output, dim=1),
            torch.clamp(F.softmax(clean_output, dim=1), min=1e-8),
        )

        clean_loss = criterion(clean_output, labels)

        loss = clean_loss + (6 * adv_loss)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(clean_output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print(
                "Train_TRADES: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                )
            )
            sys.stdout.flush()

    return losses.avg, top1.avg


def gen_adv(model, classifier, images, labels):

    model_classifier = torch.nn.Sequential(model.encoder, classifier)
    model_classifier.eval()

    device = "cuda"
    atk = PGD(model_classifier, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True)
    adv_images = atk(images.to(device), labels.to(device))

    classifier.train()
    return adv_images


def train_madry(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        adv_images = gen_adv(model, classifier, images, labels)
        model.eval()
        classifier.train()

        # compute loss
        with torch.no_grad():
            features = model.encoder(adv_images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print(
                "TrainMadry: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                )
            )
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward

            output = classifier(model.encoder(images))

            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        idx, len(val_loader), batch_time=batch_time, loss=losses, top1=top1
                    )
                )

    print(" * Acc@1 {top1.avg:.3f}".format(top1=top1))
    return losses.avg, top1.avg

def main():
    best_acc = 0
    opt = parse_option()

    start_time = time.time()
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    logger = tb_logger.Logger(logdir=opt.tb_folder)

    # training routine
    for epoch in range(1, opt.epochs + 1):

        # train for one epoch
        time1 = time.time()

        if opt.linear_loss == "Madry":
            loss, acc = train_madry(
                train_loader, model, classifier, criterion, optimizer, epoch, opt
            )
        elif opt.linear_loss == "TRADES":
            loss, acc = train_trades_linear(
                train_loader, model, classifier, criterion, optimizer, epoch, opt
            )
        else:
            loss, acc = train(train_loader, model, classifier, criterion, optimizer, epoch, opt)

        time2 = time.time()
        print(
            "Train epoch {}, total time {:.2f}, accuracy:{:.2f}".format(epoch, time2 - time1, acc)
        )

        logger.log_value("train_loss", loss, epoch)
        logger.log_value("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        logger.log_value("train_acc", acc, epoch)

        logger.log_value("time", time2 - time1, epoch)

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc

        logger.log_value("val_loss", loss, epoch)
        logger.log_value("val_acc", val_acc, epoch)

        if epoch % opt.epochs == 0:

            save_file = os.path.join(
                opt.save_folder,
                "linear_{loss}_{aug}_lr_{lr}_epoch_{epoch}_bsz_{bsz}.pth".format(
                    loss=opt.linear_loss,
                    aug=opt.aug_type,
                    lr=opt.learning_rate,
                    epoch=epoch,
                    bsz=opt.batch_size,
                ),
            )

            save_model(classifier, optimizer, opt, epoch, save_file)

    print("best accuracy: {:.2f}".format(best_acc))

    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)

if __name__ == "__main__":
    main()
