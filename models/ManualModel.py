import torch
import torch.nn as nn
import torch.nn.functional as F

class RobertaClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.manual_dense = nn.Linear(args.manual_feature_size, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.ll_proj = nn.Linear(args.hidden_size, 1)     # ll means last layer representations.

    def forward(self, manual_features):
        manual_features = manual_features.float()
        manual_features = self.manual_dense(manual_features)
        
        if self.args.activation == "tanh":
            manual_features = torch.tanh(manual_features)
        elif self.args.activation == "relu":
            manual_features = torch.relu(manual_features)

        manual_features = self.dropout(manual_features)
        proj_score = self.ll_proj(manual_features)
        return proj_score


class ManualModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(ManualModel, self).__init__()
        self.args = args
        self.classifier = RobertaClassifier(args)

    def forward(self, input_ids, input_mask, manual_features, label):
        logits = self.classifier(manual_features)

        prob = torch.sigmoid(logits)

        if self.args.loss_fct == "bce":
            loss_fct = nn.BCELoss()
        elif self.args.loss_fct == "focal":
            loss_fct = FocalLoss(self.args)
        else:
            raise ValueError("Unsupported loss function!")
        
        loss = loss_fct(prob, torch.unsqueeze(label, dim=1).float())

        return prob, loss


class FocalLoss(nn.Module):
    def __init__(self,args):
        """
        Focal Loss 实现，适用于类别不平衡的二分类问题。

        参数:
            alpha (float): 类别1的权重，范围[0, 1]。默认0.75，适用于类别1为少数类的情况。
            gamma (float): 调节难易样本的因子，默认2。
            eps (float): 数值稳定性参数，防止log(0)。
        """
        super(FocalLoss, self).__init__()
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.eps = args.eps

    def forward(self, prob, target):
        """
        参数:
            prob (Tensor): 预测的概率值（属于类别1的概率），形状为(batch_size, )
            target (Tensor): 真实标签，0或1，形状与prob相同。

        返回:
            Tensor: 计算后的Focal Loss。
        """
        # 确保输入形状一致
        prob = prob.view(-1)
        target = target.view(-1).float()  # 转换为float以匹配where的条件

        # 数值稳定性处理，防止log(0)
        prob = torch.clamp(prob, self.eps, 1 - self.eps)

        # 计算p_t：当target=1时取prob，否则取1-prob
        p_t = torch.where(target == 1, prob, 1 - prob)

        # 计算交叉熵损失项
        ce_loss = -torch.log(p_t)

        # 计算alpha因子：类别1使用alpha，类别0使用1-alpha
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)

        # 计算调制因子和总损失
        focal_loss = alpha_t * torch.pow(1 - p_t, self.gamma) * ce_loss

        # 返回平均损失
        return focal_loss.mean()


