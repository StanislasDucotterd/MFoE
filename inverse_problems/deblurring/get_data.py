import os
import torch
torch.manual_seed(0)
torch.set_grad_enabled(False)

datafolders = ['val_data', 'test_data']
kernels = [torch.load('kernels/kernel1.pth'),
           torch.load('kernels/kernel2.pth'),
           torch.load('kernels/kernel3.pth')]
blur = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=25,
                       padding=12, padding_mode='circular', bias=False)


for k, kernel in enumerate(kernels):
    blur.weight.data = kernel
    for datafolder in datafolders:
        images = os.listdir(datafolder)
        for img in images:
            x = torch.load(os.path.join(datafolder, img, 'img.pth'))
            y = blur(x)
            y_small = y + torch.randn_like(x) * 0.01
            y_medium = y + torch.randn_like(x) * 0.03
            torch.save(y_small, os.path.join(
                datafolder, img, f'y{k+1}_small_noise.pth'))
            torch.save(y_medium, os.path.join(
                datafolder, img, f'y{k+1}_medium_noise.pth'))
