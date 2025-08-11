import os
import json
import torch
from tqdm import tqdm
from dataset import H5PY
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from models.mfoe import MFoE
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
torch.manual_seed(0)


class Trainer:
    """
    """

    def __init__(self, config, device):

        self.device = device
        self.noise_val = config['noise_val']
        self.noise_range = config['noise_range']
        self.valid_epoch_num = 0

        # Datasets
        train_dataset = H5PY(config['train_dataloader']['train_data_file'])
        val_dataset = H5PY(config['train_dataloader']['val_data_file'])

        # Dataloaders
        print('Preparing the dataloaders')
        self.batch_size = config['train_dataloader']['batch_size']

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=True)
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=1)

        # Build the model
        print('Building the model')
        self.config = config
        self.model = MFoE(param_model=config['model_params'], param_multi_conv=config['multi_convolution'],
                          param_fw=config['optimization']['fixed_point_solver_fw_params'],
                          param_bw=config['optimization']['fixed_point_solver_bw_params'])
        self.model = self.model.to(device)

        print(self.model)
        print('Number of parameters in the model: ', self.model.num_params)

        # Set up the optimizer
        params = [{'params': self.model.conv_layer.parameters(), 'lr': config['training_options']['lr_conv']},
                  {'params': self.model.mu.parameters(
                  ), 'lr': config['training_options']['lr_mu']},
                  {'params': [self.model.lamb, self.model.Q_param, self.model.taus], 'lr': config['training_options']['lr_activation']}]

        self.optimizer = torch.optim.Adam(params)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1,
                                                         gamma=self.config['training_options']['lr_decay'])

        # Loss
        self.criterion = torch.nn.L1Loss(reduction='sum')
        self.scale = torch.tensor(
            self.noise_range[0]/2 + self.noise_range[1]/2).sqrt()
        self.psnr = PSNR(data_range=1.).to(device)

        # CHECKPOINTS & TENSOBOARD
        self.checkpoint_dir = os.path.join(
            config['logging_info']['log_dir'], config['exp_name'], 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        config_save_path = os.path.join(
            config['logging_info']['log_dir'], config['exp_name'], f'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(getattr(self, f'config'),
                      handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(
            config['logging_info']['log_dir'], config['exp_name'], 'tensorboard_logs')
        self.writer = tensorboard.SummaryWriter(writer_dir)

    def train(self):
        self.batch_seen = 0
        while self.batch_seen < self.config['training_options']['n_batches']:
            # train epoch
            self.train_epoch()

        self.writer.flush()
        self.writer.close()

    def train_epoch(self):
        """
        """
        self.model.train()
        tbar = tqdm(self.train_dataloader, ncols=80, position=0, leave=True)
        log = {}

        for batch_idx, data in enumerate(tbar):
            self.batch_seen += 1

            if self.batch_seen % self.config['logging_info']['log_batch'] == 0:
                self.valid_epoch()
                self.model.train()
                # SAVE CHECKPOINT
                self.save_checkpoint(self.batch_seen)

            if self.batch_seen % self.config['training_options']['n_batch_decay'] == 0:
                self.scheduler.step()

            if self.batch_seen > self.config['training_options']['n_batches']:
                break

            data = data.to(self.device)
            sigma = torch.torch.empty((data.shape[0], 1, 1, 1), device=data.device).uniform_(
                self.noise_range[0], self.noise_range[1])
            noise = sigma * torch.randn(data.shape, device=data.device)
            noisy_data = data + noise

            self.optimizer.zero_grad()
            output = self.model(noisy_data, sigma=sigma)
            loss = self.criterion(
                self.scale*output/sigma.sqrt(), self.scale*data/sigma.sqrt()) / data.shape[0]
            loss.backward()
            self.optimizer.step()

            log['loss'] = loss.item()
            log['forward_mean_iter'] = self.model.fw_niter_mean
            log['forward_max_iter'] = self.model.fw_niter_max

            self.wrt_step = self.batch_seen
            self.write_scalars_tb(log)
            tbar.set_description(
                f"T ({self.valid_epoch_num}) | TotalLoss {log['loss']:.7f}")

        return log

    def valid_epoch(self):
        self.valid_epoch_num += 1
        self.model.eval()

        with torch.no_grad():
            for noise_val in self.noise_val:
                psnr_val = 0.0
                score = 0.0
                tbar_val = tqdm(self.val_dataloader, ncols=40,
                                position=0, leave=True)
                for batch_idx, data in enumerate(tbar_val):
                    data = data.to(self.device)

                    sigma = noise_val * \
                        torch.torch.ones(
                            (data.shape[0], 1, 1, 1), device=data.device)
                    noise = sigma / 255 * \
                        torch.randn(data.shape, device=data.device)
                    noisy_data = data + noise

                    noisy_data, noise = noisy_data.to(
                        self.device), noise.to(self.device)

                    output = self.model(noisy_data, sigma=sigma/255)

                    out_val = torch.clamp(output, 0., 1.)

                    psnr_val = psnr_val + self.psnr(out_val, data)

                    data.detach()
                    noisy_data.detach()
                    out_val.detach()
                # METRICS TO TENSORBOARD
                self.wrt_mode = 'Convolutional'
                psnr_val = psnr_val/len(self.val_dataloader)
                self.writer.add_scalar(
                    f'{self.wrt_mode}/Validation PSNR sigma={noise_val}', psnr_val, self.valid_epoch_num)

    def write_scalars_tb(self, logs):
        for k, v in logs.items():
            self.writer.add_scalar(
                f'Convolutional/Training {k}', v, self.wrt_step)

    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'config': self.config
        }

        state['optimizer_state_dict'] = self.optimizer.state_dict()

        print('Saving a checkpoint:')
        # CHECKPOINTS & TENSOBOARD
        self.checkpoint_dir = os.path.join(
            self.config['logging_info']['log_dir'], self.config['exp_name'], 'checkpoints')

        filename = self.checkpoint_dir + '/checkpoint_' + str(epoch) + '.pth'
        torch.save(state, filename)
