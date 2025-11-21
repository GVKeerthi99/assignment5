import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatNet(nn.Module):
    def __init__(self, global_feat=False, transform=False):
        super(FeatNet, self).__init__()

        self.use_transform = transform
        self.return_global = global_feat

        if self.use_transform:
            self.stn1 = TransformBlock()
            self.stn2 = TransformBlock(in_channels=64)

        self.layer1 = nn.Conv1d(3, 64, 1)
        self.norm1 = nn.BatchNorm1d(64)

        self.layer2 = nn.Conv1d(64, 128, 1)
        self.norm2 = nn.BatchNorm1d(128)

        self.layer3 = nn.Conv1d(128, 1024, 1)
        self.norm3 = nn.BatchNorm1d(1024)

    def forward(self, pts):
        pts = pts.transpose(2, 1)

        if self.use_transform:
            tmat1 = self.stn1(pts)
            pts = torch.bmm(pts.transpose(2, 1), tmat1).transpose(2, 1)

        feat_local = F.relu(self.norm1(self.layer1(pts)))

        if self.use_transform:
            tmat2 = self.stn2(feat_local)
            tmp = feat_local.transpose(2, 1)
            feat_local = torch.bmm(tmp, tmat2).transpose(2, 1)

        mid_feat = F.relu(self.norm2(self.layer2(feat_local)))

        gfeat = self.norm3(self.layer3(mid_feat))
        gfeat = torch.max(gfeat, 2, keepdim=True)[0].view(-1, 1024)

        if self.return_global:
            return gfeat

        # combine
        rep_global = gfeat.unsqueeze(2).repeat(1, 1, feat_local.shape[2])

        if self.use_transform:
            return rep_global, mid_feat
        else:
            return torch.cat([rep_global, feat_local], dim=1)


class TransformBlock(nn.Module):
    def __init__(self, in_channels=3):
        super(TransformBlock, self).__init__()

        self.conv_a = nn.Conv1d(in_channels, 64, 1)
        self.conv_b = nn.Conv1d(64, 128, 1)
        self.conv_c = nn.Conv1d(128, 1024, 1)

        self.fc_a = nn.Linear(1024, 512)
        self.fc_b = nn.Linear(512, 256)
        self.fc_c = nn.Linear(256, in_channels * in_channels)

        self.bn_a = nn.BatchNorm1d(64)
        self.bn_b = nn.BatchNorm1d(128)
        self.bn_c = nn.BatchNorm1d(1024)
        self.bn_d = nn.BatchNorm1d(512)
        self.bn_e = nn.BatchNorm1d(256)

        self.in_channels = in_channels

    def forward(self, inp):
        B = inp.size(0)

        x = F.relu(self.bn_a(self.conv_a(inp)))
        x = F.relu(self.bn_b(self.conv_b(x)))
        x = F.relu(self.bn_c(self.conv_c(x)))

        x = torch.max(x, 2, keepdim=True)[0].view(B, -1)

        x = F.relu(self.bn_d(self.fc_a(x)))
        x = F.relu(self.bn_e(self.fc_b(x)))
        x = self.fc_c(x)

        eye = torch.from_numpy(
            np.eye(self.in_channels).astype(np.float32).flatten()
        ).view(1, -1).repeat(B, 1)

        if x.is_cuda:
            eye = eye.cuda()

        x = x + eye
        return x.view(B, self.in_channels, self.in_channels)


class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()

        self.fnet = FeatNet(global_feat=True)

        self.fc_a = nn.Linear(1024, 512)
        self.fc_b = nn.Linear(512, 256)
        self.fc_c = nn.Linear(256, num_classes)

        self.bn_a = nn.BatchNorm1d(512)
        self.bn_b = nn.BatchNorm1d(256)

        self.drop = nn.Dropout(p=0.3)

    def forward(self, pts):
        features = self.fnet(pts)
        x = F.relu(self.bn_a(self.fc_a(features)))
        x = F.relu(self.bn_b(self.drop(self.fc_b(x))))
        x = self.fc_c(x)
        return F.log_softmax(x, dim=1)


class seg_model(nn.Module):
    def __init__(self, num_seg_classes=6):
        super(seg_model, self).__init__()

        self.out_classes = num_seg_classes
        self.fnet = FeatNet(global_feat=False)

        self.conv_a = nn.Conv1d(1088, 512, 1)
        self.conv_b = nn.Conv1d(512, 256, 1)
        self.conv_c = nn.Conv1d(256, 128, 1)
        self.conv_d = nn.Conv1d(128, self.out_classes, 1)

        self.bn_a = nn.BatchNorm1d(512)
        self.bn_b = nn.BatchNorm1d(256)
        self.bn_c = nn.BatchNorm1d(128)

    def forward(self, pts):
        B, N, _ = pts.shape

        x = self.fnet(pts)

        x = F.relu(self.bn_a(self.conv_a(x)))
        x = F.relu(self.bn_b(self.conv_b(x)))
        x = F.relu(self.bn_c(self.conv_c(x)))

        x = self.conv_d(x)
        x = F.log_softmax(
            x.transpose(2, 1).contiguous().view(-1, self.out_classes),
            dim=-1
        )
        return x.view(B, N, self.out_classes)


class trans_cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(trans_cls_model, self).__init__()

        self.fnet = FeatNet(global_feat=True, transform=True)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.drop = nn.Dropout(0.3)

    def forward(self, pts):
        x = self.fnet(pts)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.drop(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class trans_seg_model(nn.Module):
    def __init__(self, num_seg_classes=6):
        super(trans_seg_model, self).__init__()

        self.num_seg = num_seg_classes
        self.fnet = FeatNet(transform=True)

        self.cv1 = nn.Conv1d(1024, 512, 1)
        self.cv2 = nn.Conv1d(512, 256, 1)
        self.cv3 = nn.Conv1d(256, 128, 1)
        self.cv4 = nn.Conv1d(256, self.num_seg, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, pts):
        B, N, _ = pts.size()

        x, skip_feat = self.fnet(pts)

        x = F.relu(self.bn1(self.cv1(x)))
        x = F.relu(self.bn2(self.cv2(x)))
        x = F.relu(self.bn3(self.cv3(x)))

        x = torch.cat([skip_feat, x], dim=1)
        x = self.cv4(x)

        x = F.log_softmax(
            x.transpose(2, 1).contiguous().view(-1, self.num_seg),
            dim=-1
        )
        return x.view(B, N, self.num_seg)
