import torch
import warnings
import sys
from utils.AdEMAMix import AdEMAMix
from utils.adopt import ADOPT

def get_optimizer(optimizer_choice, model, lr=1e-4):
    if optimizer_choice == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_choice == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    elif optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    elif optimizer_choice == 'ademamix':
        optimizer = AdEMAMix(model.parameters(), lr=lr)
    elif optimizer_choice == 'adopt':
        optimizer = ADOPT(model.parameters(), lr=lr, decouple=True)
    else: sys.exit('please choose a valid optimizer')

    return optimizer
