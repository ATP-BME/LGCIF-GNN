import torch

import torch.nn as nn
import torch.nn.functional as F

def inner_loss(label, matrixs):

    loss = 0

    if torch.sum(label == 0) > 1:
        loss += torch.mean(torch.var(matrixs[label == 0], dim=0))

    if torch.sum(label == 1) > 1:
        loss += torch.mean(torch.var(matrixs[label == 1], dim=0))

    return loss


def intra_loss(label, matrixs):
    a, b = None, None

    if torch.sum(label == 0) > 0:
        a = torch.mean(matrixs[label == 0], dim=0)

    if torch.sum(label == 1) > 0:
        b = torch.mean(matrixs[label == 1], dim=0)
    if a is not None and b is not None:
        return 1 - torch.mean(torch.pow(a-b, 2))
    else:
        return 0


def mixup_cluster_loss(matrixs, y_a, y_b, lam, intra_weight=2):

    y_1 = lam * y_a.float() + (1 - lam) * y_b.float()

    y_0 = 1 - y_1

    bz, roi_num, _ = matrixs.shape
    matrixs = matrixs.reshape((bz, -1))
    sum_1 = torch.sum(y_1)
    sum_0 = torch.sum(y_0)
    loss = 0.0

    if sum_0 > 0:
        center_0 = torch.matmul(y_0, matrixs)/sum_0
        diff_0 = torch.norm(matrixs-center_0, p=1, dim=1)
        loss += torch.matmul(y_0, diff_0)/(sum_0*roi_num*roi_num)
    if sum_1 > 0:
        center_1 = torch.matmul(y_1, matrixs)/sum_1
        diff_1 = torch.norm(matrixs-center_1, p=1, dim=1)
        loss += torch.matmul(y_1, diff_1)/(sum_1*roi_num*roi_num)
    if sum_0 > 0 and sum_1 > 0:
        loss += intra_weight * \
            (1 - torch.norm(center_0-center_1, p=1)/(roi_num*roi_num))

    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, 1] where N is the batch size and 1 is the number of classes
        # targets: [N] where each element is 0 or 1
        # BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss