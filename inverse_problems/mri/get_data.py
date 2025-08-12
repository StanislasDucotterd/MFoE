import os
import torch
from fastmri.data import subsample
torch.manual_seed(0)


def center_crop(data, shape):
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


datafolders = ['val_data', 'test_data']
img_types = ['pd', 'pdfs']
cf_accs = [(4, 0.08), (8, 0.04)]
noise_std_dev = 0.01

for cf_acc in cf_accs:
    acceleration, center_fraction = cf_acc
    mask_func = subsample.RandomMaskFunc(
        center_fractions=[center_fraction], accelerations=[acceleration])
    folder = 'singlecoil_acc_' + \
        str(acceleration) + '_cf_' + str(center_fraction) + \
        '_noisesd_' + str(noise_std_dev)
    for datafolder in datafolders:
        for img_type in img_types:
            images = os.listdir(os.path.join(datafolder, img_type))
            for image in images:
                path = os.path.join(datafolder, folder, img_type, image)
                if not os.path.exists(path):
                    os.makedirs(path)
                x = torch.load(os.path.join(
                    datafolder, img_type, image, 'x.pt'))
                x_crop = center_crop(x, (320, 320))
                clean_y = torch.fft.fft2(x, norm='ortho')
                y = clean_y + noise_std_dev * torch.randn_like(clean_y)
                h = y.shape[2]
                w = y.shape[3]
                mask, num_low_frequencies = mask_func([h, w, 1])
                mask = mask[:, :, 0]
                mask = mask.expand(h, -1)
                mask = mask[None, None, ...]
                mask = torch.fft.ifftshift(mask, dim=(2, 3))
                torch.save(x_crop, path + '/x_crop.pt')
                torch.save(y, path + '/y.pt')
                torch.save(mask, path + '/mask.pt')
