import torch
from torch import nn
from torchvision import models, transforms

# transforms
vgg16_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model class
def decoder_block(in_channel, out_channel, inter=None):
    inter = inter if inter else out_channel

    block = nn.Sequential(
        nn.Conv2d(in_channel, inter, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(inter, inter, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(inter, out_channel, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )

    return block

class UNetVGG16(nn.Module):
    """
    Creates UNet architecture with VGG16 as encoder.
    Related links:
        http://doi.org/10.12928/telkomnika.v18i3.14753
        https://github.com/zhoudaxia233/PyTorch-Unet/blob/master/vgg_unet.py

    Args:
    input_shape: int | indicating number of input channels.
    output_shape: int | indicating number of output units.
    """

    def __init__(self, out_channel):
        super().__init__()
        self.encoder = models.vgg16(weights='DEFAULT').features
        self.enc1 = nn.Sequential(*self.encoder[:4])                  # [1, 64, 224, 224]
        self.enc2 = nn.Sequential(*self.encoder[4:9])                 # [1, 128, 112, 112]
        self.enc3 = nn.Sequential(*self.encoder[9:16])                # [1, 256, 56, 56]
        self.enc4 = nn.Sequential(*self.encoder[16:23])               # [1, 512, 28, 28]
        self.enc5 = nn.Sequential(*self.encoder[23:30])               # [1, 512, 14, 14]
        del self.encoder

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.dec1 = decoder_block(1024, 256)
        self.dec2 = decoder_block(512 + 256, 128, 256)
        self.dec3 = decoder_block(256 + 128, 64, 128)
        self.dec4 = decoder_block(128 + 64, 32, 64)

        self.dec5 = nn.Sequential(
            nn.Conv2d(64 + 32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channel, kernel_size=3, padding=1)
        )

    def forward(self, x):
        patch1 = self.enc1(x)
        patch2 = self.enc2(patch1)
        patch3 = self.enc3(patch2)
        patch4 = self.enc4(patch3)
        patch5 = self.enc5(patch4)

        x = self.bottleneck(patch5)

        x = torch.cat([x, patch5], dim=1)
        x = self.dec1(x)
        x = torch.cat([x, patch4], dim=1)
        x = self.dec2(x)
        x = torch.cat([x, patch3], dim=1)
        x = self.dec3(x)
        x = torch.cat([x, patch2], dim=1)
        x = self.dec4(x)
        x = torch.cat([x, patch1], dim=1)
        x = self.dec5(x)

        return x
