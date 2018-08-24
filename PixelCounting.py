import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelCountingRegress(nn.Module):

    def __init__(self):
        super(PixelCountingRegress, self).__init__()


    def forward(self, x, y):

        shared_pixels = torch.sum(torch.mul(x, y))
        gt_pixels = torch.sum(y)
        loss = torch.tensor(1.0)
        return loss - torch.div(shared_pixels, gt_pixels)


