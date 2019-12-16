import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns 
import collections
import functools

# Use this file to generate CNN stuff for default model classes 

# Train last layer on high learning rate 
# Train whole thing on lower learning rate 

class ImageNet_Model_Wrapper(nn.Module): 
    def __init__(self, model_name, freeze_all_but_last=True, embedding_sz=512): 
        super().__init__()
        self.model_choices = frozenset(['resnet', 'vgg', 'densenet'])
        self.model = None 
        self.freeze = freeze_all_but_last
        # define self.model to be as specified 
        if model_name not in self.model_choices: 
            raise ValueError(f"Invalid model_name choice: \'{model_name}\'; " + 
                             f"valid model_names: {list(self.model_choices)}")
        elif model_name == 'resnet': 
            self.model = torchvision.models.resnet50(pretrained=True)
            # modify self.model to fit parameters 
            linear_inpt = self.model.fc.in_features
            self.model.fc = nn.Linear(linear_inpt, embedding_sz)
            if self.freeze: 
                self.__freeze_fc_child_ty_model("fc")
        elif model_name == "vgg": 
            self.model = torchvision.models.vgg16(pretrained=True)
            linear_inpt = self.model.classifier[6].in_features
            new_classifier = list(self.model.classifier.children())
            new_classifier[-1] = nn.Linear(linear_inpt, embedding_sz)
            self.model.classifier = nn.Sequential(*new_classifier)
            if self.freeze: 
                for _, c in self.model.named_children(): 
                    for param_name, p in c.named_parameters(): 
                        if not (param_name == "6.weight" or param_name == "6.bias"): 
                            p.requires_grad = False 
        elif model_name == "densenet": 
            self.model = torchvision.models.densenet121(pretrained=True)
            linear_inpt = self.model.classifier.in_features
            self.model.classifier = nn.Linear(linear_inpt, embedding_sz)
            if self.freeze:
                self.__freeze_fc_child_ty_model("classifier")
        # print(self.model)

    def unfreeze(self): 
        for param in self.model.parameters(): 
            param.requires_grad = True

    # https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    def print_trainable_params(self):
        total_params = sum([param.numel() for param in self.model.parameters()]) 
        trainable_params = sum([param.numel() for param in self.model.parameters() if param.requires_grad])
        print(f'Trainable Params: {trainable_params}, {100 * trainable_params/total_params}% of Total Parameters: {total_params}')
 
    def __freeze_fc_child_ty_model(self, last_layer_name):
        for name, child in self.model.named_children():
            if name == last_layer_name: 
                break 
            for _, params in child.named_parameters():  
                params.requires_grad = False 

    def forward(self, x): 
        return self.model(x)

inpt = torch.randn(1, 3, 64, 64) 
model = ImageNet_Model_Wrapper('densenet', freeze_all_but_last=True)
model.print_trainable_params()
outpt = model(inpt)
print(outpt.shape)
