import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import cls_model, seg_model, trans_cls_model, trans_seg_model
from data_loader import get_data_loader
from utils import save_checkpoint, create_dir


def train(loader, net, optimizer, epoch_idx, args, writer):

    net.train()
    base_step = epoch_idx * len(loader)
    total_loss = 0

    for idx, batch in enumerate(loader):
        pts, lbl = batch
        pts = pts.to(args.device)
        lbl = lbl.to(args.device).long()

        # ------ TO DO: Forward Pass ------
        out_logits = net(pts)
        outputs = out_logits

        if args.task in ["seg", "trans_seg"]:
            lbl = lbl.view(-1)
            outputs = outputs.view(-1, args.num_seg_class)

        criterion = torch.nn.CrossEntropyLoss()
        loss_val = criterion(outputs, lbl)
        total_loss += loss_val

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        writer.add_scalar("train_loss", loss_val.item(), base_step + idx)

    return total_loss


def test(loader, net, epoch_idx, args, writer):

    net.eval()

    if args.task in ["cls", "trans_cls"]:
        correct = 0
        total = 0
        for pts, lbl in loader:
            pts = pts.to(args.device)
            lbl = lbl.to(args.device).long()

            # ------ TO DO: Make Predictions ------
            with torch.no_grad():
                scores = net(pts)
                preds = scores.max(dim=1)[1]

            correct += preds.eq(lbl).cpu().sum().item()
            total += lbl.size(0)

        acc = correct / total

    else:
        correct = 0
        total = 0
        for pts, lbl in loader:
            pts = pts.to(args.device)
            lbl = lbl.to(args.device).long()

            # ------ TO DO: Make Predictions ------
            with torch.no_grad():
                scores = net(pts)
                preds = scores.max(dim=2)[1]

            correct += preds.eq(lbl).cpu().sum().item()
            total += lbl.view(-1, 1).size(0)

        acc = correct / total

    writer.add_scalar("test_acc", acc, epoch_idx)
    return acc


def main(args):

    create_dir(args.checkpoint_dir)
    create_dir('./logs')

    writer = SummaryWriter('./logs/{}'.format(args.task + "_" + args.exp_name))

    # ------ TO DO: Initialize Model ------
    if args.task == "cls":
        net = cls_model()
    elif args.task == "trans_cls":
        net = trans_cls_model()
    elif args.task == "trans_seg":
        net = trans_seg_model()
    else:
        net = seg_model(num_seg_classes=args.num_seg_class)

    net = net.to(args.device)

    if args.load_checkpoint:
        ckpt = "{}/{}.pt".format(args.checkpoint_dir, args.load_checkpoint)
        with open(ckpt, 'rb') as f:
            state_dict = torch.load(f, map_location=args.device)
            net.load_state_dict(state_dict)
        print("successfully loaded checkpoint from {}".format(ckpt))

    optimizer = optim.Adam(net.parameters(), args.lr, betas=(0.9, 0.999))

    train_loader = get_data_loader(args=args, train=True)
    test_loader = get_data_loader(args=args, train=False)

    print("successfully loaded data")

    best_acc = -1

    print("======== start training for {} task ========".format(args.task))
    print("(check tensorboard for plots of experiment logs/{})".format(args.task + "_" + args.exp_name))

    for ep in range(args.num_epochs):

        train_loss = train(train_loader, net, optimizer, ep, args, writer)
        acc = test(test_loader, net, ep, args, writer)

        print("epoch: {}   train loss: {:.4f}   test accuracy: {:.4f}".format(ep, train_loss, acc))

        if ep % args.checkpoint_every == 0:
            print("checkpoint saved at epoch {}".format(ep))
            save_checkpoint(epoch=ep, model=net, args=args, best=False)

        if acc >= best_acc:
            best_acc = acc
            print("best model saved at epoch {}".format(ep))
            save_checkpoint(epoch=ep, model=net, args=args, best=True)

    print("======== training completes ========")


def create_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default="cls")
    parser.add_argument('--num_seg_class', type=int, default=6)

    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--exp_name', type=str, default="exp")

    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_every', type=int, default=10)

    parser.add_argument('--load_checkpoint', type=str, default='')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.checkpoint_dir = args.checkpoint_dir + "/" + args.task

    main(args)
