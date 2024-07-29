import torch
import torch.nn as nn
import torch.nn.functional as F

class BNHead(nn.Module):
    def __init__(self, in_channels, num_classes, ignore_index=255, align_corners=False):
        super(BNHead, self).__init__()
        self.input_transform = 'resize_concat'
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        
        self.loss_decode = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1)
        self.bn = nn.SyncBatchNorm(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.conv_seg(x)
        return x