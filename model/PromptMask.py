from config.config import path, templete
from pytorch_transformers import BertForMaskedLM
import torch
import torch.nn as nn

class PromptMask(nn.Module):
    def __init__(self):
        super(PromptMask, self).__init__()
        self.roberta = BertForMaskedLM.from_pretrained(path['pretrained'])

    def forward(self, input_x):
        mask0 = (input_x == 103)
        mask1 = (input_x != 0).type(torch.long)
        input_x = self.roberta(input_x, attention_mask=mask1)
        x = input_x[0][mask0]
        return x