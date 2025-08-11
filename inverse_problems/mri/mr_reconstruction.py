import os
import sys
import math
import types
import torch
import argparse
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
torch.set_grad_enabled(False)
torch.manual_seed(0)

sys.path.append('../..')
from models.mfoe import MFoE
from models.optimization import SAGD
from inverse_problems.tune_hyperparameters import tune_hyperparameters


def center_crop(data, shape):
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def test_hyperparameter(lamb, sigma):
    mean_psnr = 0.
    if testing:
        times = []
        starter, ender = torch.cuda.Event(
            enable_timing=True), torch.cuda.Event(enable_timing=True)
    for image in images:
        mask = torch.load(os.path.join(data_folder, image,
                          'mask.pt'), weights_only=True).to(device)
        x_gt = torch.load(os.path.join(data_folder, image,
                          'x_crop.pt'), weights_only=True).to(device)
        y = torch.load(os.path.join(data_folder, image, 'y.pt'),
                       weights_only=True).to(device)
        Hty = torch.fft.ifft2(y*mask, norm='ortho').real

        def reconstruct(self, x, y, sigma):
            grad, cost = self.grad_cost(x, sigma)
            grad = 1. / (1. + self.lamb.exp()) * (torch.fft.ifft2(torch.fft.fft2(x,
                                                                                 norm='ortho')*mask, norm='ortho').real - Hty + grad)
            cost = cost + (1/2)*((torch.fft.fft2(x, norm='ortho') - y)
                                 * mask).norm(dim=(1, 2, 3), p=2)**2
            return grad, cost
        model.reconstruct = types.MethodType(reconstruct, model)

        model.lamb.data = model.lamb + math.log(lamb)
        if testing:
            starter.record()
        pred = SAGD(Hty, y, model, sigma * torch.ones(1, 1,
                    1, 1, device=device), max_iter, tol)[0]
        if testing:
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender) / 1000)
            torch.save(pred.cpu(), os.path.join(
                data_folder, image, 'pred_mfoe.pt'))
        model.lamb.data = model.lamb - math.log(lamb)
        mean_psnr += psnr(center_crop(pred, (320, 320)),
                          x_gt).item() / len(images)
    if testing:
        print(f'Mean time per image: {torch.tensor(times).mean():.4f} seconds')
        print(f'Standard deviation: {torch.tensor(times).std():.4f} seconds')
    return mean_psnr


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-d', '--device', default="cpu",
                    type=str, help='device to use')
device = parser.parse_args().device
psnr = PSNR(data_range=1.).to(device)

data_type = 'singlecoil_acc_4_cf_0.08_noisesd_0.01/pdfs/'

model_name = 'MFoE_groupsize_4'
infos = torch.load('../../trained_models/' + model_name +
                   '/checkpoints/checkpoint_5000.pth', map_location='cpu', weights_only=True)
config = infos['config']
model = MFoE(param_model=config['model_params'],
             param_multi_conv=config['multi_convolution'],
             param_fw=config['optimization']['fixed_point_solver_fw_params'],
             param_bw=config['optimization']['fixed_point_solver_bw_params'])
model.load_state_dict(infos['state_dict'], strict=False)
model = model.to(device)
model.eval()

print(" **** Updating the Lipschitz constant **** ")
model.conv_layer.spectral_norm(mode="power_method", n_steps=500)

tol = 1e-5
max_iter = 1000
best_lambda = 0.1
best_sigma = 0.1

testing = False
data_folder = os.path.join('val_data', data_type)
images = os.listdir(data_folder)
best_lambda, best_sigma = tune_hyperparameters(
    test_hyperparameter, best_lambda, best_sigma)

testing = True
data_folder = os.path.join('test_data', data_type)
images = os.listdir(data_folder)
final_psnr = test_hyperparameter(best_lambda, best_sigma)

print(data_folder)
print(model_name)
print(
    f'Best PSNR is {final_psnr:.6} with Lambda={best_lambda:.6} and Sigma={best_sigma:.6}')
