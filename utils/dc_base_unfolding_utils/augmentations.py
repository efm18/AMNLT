import torch
import torch.nn as nn
from torchvision.transforms import v2


class AugmentStage(nn.Module):
    def __init__(self):
        super(AugmentStage, self).__init__()
        self.transforms = v2.Compose(
            [
                v2.RandomApply([v2.ElasticTransform(alpha=25.0)], p=0.2),
                v2.RandomRotation(degrees=3, p=0.2),
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.2,
                ),
                v2.RandomApply([v2.GaussianBlur(kernel_size=23)], p=0.2),
            ]
        )

    def forward(self, x):
        with torch.no_grad():
            return torch.stack([self.transforms(image) for image in x])
