import torch
import torch.nn as nn
import torch.nn.functional as F


def featurize(input_img, model, device):
    '''Helper method for running an image patch through a model.

    Args:
        input_img (np.ndarray): Image in (C x H x W) format with a dtype of uint8.
        model (torch.nn.Module): Feature extractor network
    '''
    assert len(input_img.shape) == 3
    input_img = torch.from_numpy(input_img / 255.).float()
    input_img = input_img.to(device)
    with torch.no_grad():
        feats = model(input_img.unsqueeze(0)).cpu().numpy()
    return feats


class RCF(nn.Module):

    def __init__(self, num_filters=16, patch_size=3, num_channels=4, bias=-1.0):
        super(RCF,self).__init__()

        assert num_filters % 2 == 0

        self.conv1 = nn.Conv2d(
            num_channels, num_filters // 2,
            kernel_size=patch_size,
            stride=1, padding=0, dilation=1,
            bias=True
        )

        nn.init.normal_(self.conv1.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.conv1.bias, bias)

    def forward(self,inputs):
        x1a = F.relu(self.conv1(inputs), inplace=True)
        x1b = F.relu(-self.conv1(inputs), inplace=True)

        x1a = F.adaptive_avg_pool2d(x1a, (1,1)).squeeze()
        x1b = F.adaptive_avg_pool2d(x1b, (1,1)).squeeze()

        if len(x1a.shape) == 1: # case where we passed a single input
            return torch.cat((x1a, x1b), dim=0)
        elif len(x1a.shape) == 2: # case where we passed a batch of > 1 inputs
            return torch.cat((x1a, x1b), dim=1)