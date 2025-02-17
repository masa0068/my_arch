from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug
import numpy as np
import torch

import wandb

class Mixup(ContinualModel):
    NAME = 'mixup'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_lodaer, transform):
        super(Mixup, self).__init__(backbone, loss, args, len_train_lodaer, transform)
        self.buffer = Buffer(self.args.model.buffer_size, self.device)

    def observe(self, inputs1, inputs2, notaug_inputs=None, labels=None):

        self.opt.zero_grad()
        data_dict, feature = self.net.forward(inputs1.to(self.device, non_blocking=True), inputs2.to(self.device, non_blocking=True))

        loss = data_dict['loss'].mean()
        data_dict['loss'] = data_dict['loss'].mean()

        try:#wandb#################
            wandb.log({"loss":loss})
        except:
            pass
        ###########################
            
        loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.args.train.base_lr})

        return data_dict
