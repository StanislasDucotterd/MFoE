
import os
import torch
from ct_forward_utils import get_operators
torch.set_grad_enabled(False)
torch.manual_seed(0)

nums_angles = [20, 40, 60]
datafolders = ['val_data', 'test_data']

img_size = 362
space_range = 1
det_shape = 256

for num_angles in nums_angles:
    fwd_op, fbp_op, bp_op = get_operators(img_size=img_size, space_range=space_range, num_angles=num_angles,
                                          det_shape=det_shape, fix_scaling=True)
    for datafolder in datafolders:
        files = os.listdir(datafolder)

        for file in files:
            x = torch.load(os.path.join(datafolder, file, 'image.pth'))
            x = x
            y = fwd_op(x)
            y += torch.randn_like(y) * 0.1
            torch.save(y, os.path.join(datafolder, file,
                       f'y_{num_angles}_{det_shape}.pth'))
