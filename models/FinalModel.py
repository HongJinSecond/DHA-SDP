from torch import nn
from peft import PeftModel
import torch

class FinalModel(nn.Module):
    def __init__(self, args,base_model,cluster_manager=None):
        super(FinalModel, self).__init__()
        self.args = args
        self.base_model = base_model
        self.adapter_model = base_model
        self.cluster_manager = cluster_manager
        self.prefix="test"

    def forward(self, input_ids, input_mask, manual_features, label,use_base=False,outlier_mask=None):
        if outlier_mask==None:
            if use_base:
                with self.adapter_model.disable_adapter():
                    prob, loss = self.adapter_model.get_base_model()(input_ids, input_mask, manual_features, label)

            else:
                prob, loss = self.adapter_model(input_ids, input_mask, manual_features, label)
            return prob, loss
        else:
            # outlier_mask == True means the sample is outlier. use base model
            prob=torch.zeros(size=(len(input_ids),1),dtype=torch.float32,device=self.args.device)
            loss_1=None
            loss_2=None
            
            with self.adapter_model.disable_adapter():
                prob[outlier_mask],loss_1=self.adapter_model.get_base_model()(input_ids[outlier_mask], input_mask[outlier_mask], manual_features[outlier_mask], label[outlier_mask])
            prob[~outlier_mask],loss_2= self.adapter_model(input_ids[~outlier_mask], input_mask[~outlier_mask], manual_features[~outlier_mask], label[~outlier_mask])

            return prob, loss_1*torch.sum(outlier_mask)+loss_2*(1-torch.sum(outlier_mask))

    def load_lora(self,loras_path,lora_name=None):
        if isinstance(self.adapter_model,PeftModel):
            self.adapter_model.load_adapter(loras_path,adapter_name=self.prefix+lora_name)
        else:
            self.adapter_model = PeftModel.from_pretrained(self.base_model,loras_path,adapter_name=self.prefix+lora_name)

    def change_loras(self,lora_name):
        self.adapter_model.set_adapter(self.prefix+lora_name)


class FinalModelWeight(nn.Module):
    def __init__(self, args, base_model, cluster_manager=None):
        super(FinalModelWeight, self).__init__()
        self.args = args
        self.base_model = base_model
        self.adapter_model = base_model
        self.cluster_manager = cluster_manager
        self.prefix = "test"

    def forward(self, input_ids, input_mask, manual_features, label, use_base=False, outlier_mask=None,weight=0.1):
        if outlier_mask == None:
            if use_base:
                with self.adapter_model.disable_adapter():
                    prob, loss = self.adapter_model.get_base_model()(input_ids, input_mask, manual_features, label)

            else:
                prob, loss = self.adapter_model(input_ids, input_mask, manual_features, label)
            return prob, loss
        else:
            # outlier_mask == True means the sample is outlier. use base model
            prob = torch.zeros(size=(len(input_ids), 1), dtype=torch.float32, device=self.args.device)

            with self.adapter_model.disable_adapter():
                prob_base, loss_l1 = self.adapter_model.get_base_model()(input_ids[outlier_mask],
                                                                                 input_mask[outlier_mask],
                                                                                 manual_features[outlier_mask],
                                                                                 label[outlier_mask])
            prob_lora, loss_l2 = self.adapter_model(input_ids[outlier_mask],
                                                                             input_mask[outlier_mask],
                                                                             manual_features[outlier_mask],
                                                                             label[outlier_mask])
            prob[outlier_mask] = prob_base*(1-weight)+prob_lora*weight
            loss_1=loss_l1*(1-weight)+loss_l2*weight
            if (~outlier_mask).sum()!=0: 
                #if ~outlier_mask.sum()==0, that means all the samples are outliers and do not need to run the following case.
                prob[~outlier_mask], loss_2 = self.adapter_model(input_ids[~outlier_mask], input_mask[~outlier_mask],
                                                                    manual_features[~outlier_mask], label[~outlier_mask])
                prob = torch.clamp(prob, 1e-7, 1 - 1e-7)

                return prob, loss_1 * torch.sum(outlier_mask) + loss_2 * (1 - torch.sum(outlier_mask))
            
            else:
                return prob, loss_1

    def load_lora(self, loras_path, lora_name=None):
        if isinstance(self.adapter_model, PeftModel):
            self.adapter_model.load_adapter(loras_path, adapter_name=self.prefix + lora_name)
        else:
            self.adapter_model = PeftModel.from_pretrained(self.base_model, loras_path,
                                                           adapter_name=self.prefix + lora_name)

    def change_loras(self, lora_name):
        self.adapter_model.set_adapter(self.prefix + lora_name)


class FinalModelCCT5(nn.Module):
    def __init__(self, args,base_model,cluster_manager=None):
        super(FinalModelCCT5, self).__init__()
        self.args = args
        self.base_model = base_model
        self.adapter_model = base_model
        self.cluster_manager = cluster_manager
        self.prefix="test"

    def forward(self, input_ids, input_mask, manual_features, label,use_base=False,outlier_mask=None,weight=0.1):
        if outlier_mask==None:
            if use_base:
                with self.adapter_model.disable_adapter():
                    logits,loss = self.adapter_model.get_base_model()(
                        cls=True,
                        input_ids=input_ids,
                        manual_feature=manual_features,
                        labels=label,
                        attention_mask=input_mask
                    )
                    prob = torch.nn.functional.softmax(
                            logits, dim=1)[:, 1].reshape(-1,1)
            else:
                logits, loss = self.adapter_model(
                        cls=True,
                        input_ids=input_ids,
                        manual_feature=manual_features,
                        labels=label,
                        attention_mask=input_mask
                    )
                prob = torch.nn.functional.softmax(
                            logits, dim=1)[:, 1].reshape(-1,1)
            return prob, loss
        
        else:
            # outlier_mask == True means the sample is outlier. use base model
            prob=torch.zeros(size=(len(input_ids),1),dtype=torch.float32,device=self.args.device)

            with self.adapter_model.disable_adapter():
                logits_base,loss_base= self.adapter_model.get_base_model()(
                        cls=True,
                        input_ids=input_ids[outlier_mask],
                        manual_feature=manual_features[outlier_mask],
                        labels=label[outlier_mask],
                        attention_mask=input_mask[outlier_mask]
                    )
                prob_base = torch.nn.functional.softmax(
                        logits_base, dim=1)[:, 1].reshape(-1,1)
                
            logits_lora, loss_lora = self.adapter_model(
                    cls=True,
                    input_ids=input_ids[outlier_mask],
                    manual_feature=manual_features[outlier_mask],
                    labels=label[outlier_mask],
                    attention_mask=input_mask[outlier_mask]
            )
            prob_lora = torch.nn.functional.softmax(
                    logits_lora, dim=1)[:, 1].reshape(-1,1)
            prob[outlier_mask] = prob_base*(1-weight)+prob_lora*weight
            loss_1=loss_base*(1-weight)+loss_lora*weight


            if (~outlier_mask).sum()!=0: 
                #if ~outlier_mask.sum()==0, that means all the samples are outliers and do not need to run the following case.
                logits2, loss_2 = self.adapter_model(
                    cls=True,
                    input_ids=input_ids[~outlier_mask],
                    manual_feature=manual_features[~outlier_mask],
                    labels=label[~outlier_mask],
                    attention_mask=input_mask[~outlier_mask]
                )
                prob2 = torch.nn.functional.softmax(
                        logits2, dim=1)[:, 1].reshape(-1,1)
                prob2 = torch.clamp(prob2, 1e-7, 1 - 1e-7)
                prob[~outlier_mask]=prob2
                return prob, loss_1 * torch.sum(outlier_mask) + loss_2 * (1 - torch.sum(outlier_mask))
            
            else:
                return prob, loss_1
  

    def load_lora(self,loras_path,lora_name=None):
        if isinstance(self.adapter_model,PeftModel):
            self.adapter_model.load_adapter(loras_path,adapter_name=self.prefix+lora_name)
        else:
            self.adapter_model = PeftModel.from_pretrained(self.base_model,loras_path,adapter_name=self.prefix+lora_name)

    def change_loras(self,lora_name):
        self.adapter_model.set_adapter(self.prefix+lora_name)


