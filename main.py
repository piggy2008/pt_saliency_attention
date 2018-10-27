import torch
import torch.nn as nn
import torch.nn.functional as F
from image_data_loader import ImageAndPriorSeqData
import time
from models_base import VideoSaliency


if __name__ == '__main__':

    model = VideoSaliency()
    count = 0
    for child in model.named_children():
        if child[0].find('fc') >= 0:
            print(child[0] + ' not froze')
            continue
        else:
            print (child[0] + ' froze')
            for param in child[1].parameters():
                param.requires_grad = False