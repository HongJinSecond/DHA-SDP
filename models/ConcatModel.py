import torch
import torch.nn as nn
import torch.nn.functional as F

class RobertaClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.manual_dense = nn.Linear(args.manual_feature_size, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.cat_proj = nn.Linear(2 * args.hidden_size, 1)

    def forward(self, features, manual_features):
        cls_features = features[:, 0, :]
        manual_features = manual_features.float()
        manual_features = self.manual_dense(manual_features)
        if self.args.activation == "tanh":
            manual_features = torch.tanh(manual_features)
        elif self.args.actibation == "relu":
            manual_features = torch.relu(manual_features)

        cat_features = torch.cat((cls_features, manual_features), dim=1)
        cat_features = self.dropout(cat_features)
        proj_score = self.cat_proj(cat_features)
        return proj_score


class ConcatModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(ConcatModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = RobertaClassifier(args)

    def forward(self, input_ids, input_mask, manual_features, label):
        if self.args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
            if self.args.base_train:
                outputs = self.encoder.base_model.model(input_ids=input_ids, attention_mask=input_mask)
            else:
                outputs = self.encoder(input_ids=input_ids, attention_mask=input_mask)
        elif self.args.pretrained_model in ["codet5"]:
            outputs = self.encoder.encoder(input_ids=input_ids, attention_mask=input_mask)
        elif self.args.pretrained_model in ["plbart", "plbart-large"]:
            if self.args.base_train:
                outputs = self.encoder.base_model.model.model.encoder(input_ids=input_ids, attention_mask=input_mask)
            else:
                outputs = self.encoder.model.encoder(input_ids=input_ids, attention_mask=input_mask)

        logits = self.classifier(outputs[0], manual_features)

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
        Implementation of Focal Loss for imbalanced binary classification.
        
        Args:
            alpha (float): Weight for class 1, in range [0, 1]. Default 0.75, suitable when class 1 is the minority.
            gamma (float): Focusing parameter to adjust easy/hard samples. Default 2.
            eps (float): Numerical stability term to prevent log(0).
        """
        super(FocalLoss, self).__init__()
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.eps = args.eps

    def forward(self, prob, target):
        """
        Args:
            prob (Tensor): Predicted probability for class 1, shape (batch_size, )
            target (Tensor): Ground truth labels (0 or 1), same shape as prob.
            
        Returns:
            Tensor: Computed Focal Loss value.
        """
        # Ensure consistent shapes
        prob = prob.view(-1)
        target = target.view(-1).float()  # Convert to float for torch.where condition
        
        # Numerical stability: prevent log(0)
        prob = torch.clamp(prob, self.eps, 1 - self.eps)
        
        # Compute p_t: prob when target=1, 1-prob otherwise
        p_t = torch.where(target == 1, prob, 1 - prob)
        
        # Cross-entropy term
        ce_loss = -torch.log(p_t)
        
        # Alpha factor: alpha for class 1, 1-alpha for class 0
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # Compute modulating factor and final loss
        focal_loss = alpha_t * torch.pow(1 - p_t, self.gamma) * ce_loss
        
        # Return mean loss
        return focal_loss.mean()


