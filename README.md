# Multivariate Fields of Experts
Implementation of experiments done in : https://arxiv.org/abs/2508.06490v1

![alt text](https://github.com/StanislasDucotterd/MFoE/blob/main/potential.png?raw=true)

#### Description
We introduce the multivariate fields of experts, a new framework for the learning of image priors. Our model generalizes existing fields of experts methods by incorporating multivariate potential functions constructed via Moreau envelopes of the ℓ∞-norm. We demonstrate the effectiveness of our proposal across a range of inverse problems that include image denoising, deblurring, compressed-sensing magnetic-resonance imaging, and computed tomography. The proposed approach outperforms comparable univariate models and achieves performance close to that of deep-learning-based  regularizers while being significantly faster, requiring fewer parameters, and being trained on substantially fewer data. In addition, our model retains a relatively high level of interpretability due to its structured design. 

#### Requirements
The required packages:
- `pytorch`
- `torchdeq`
- `matplotlib`
- `fastmri` (to generate the MRI masks)
- `astra-toolbox` (for CT reconstruction)
- `odl` (for CT reconstruction)

The envs folder contains two files; `ct_reconstruction.yml` describes the environment used for the CT reconstruction, `training.yml` describes the envionment used for everything else.

#### Training

You can train a model with the following command:

```bash
python train.py --device cpu or cuda:n
```

#### Config file details️

Information about the hyperparameters that yield the best performance for the four experiments can be found in the config folder. 

Below we detail the model informations that can be controlled in the file `config.json`.

```javascript
{
    "exp_name": "MFoE_groupsize_4",
    "logging_info": {
        "log_batch": 5000,
        "log_dir": "trained_models/" 
    },
    "model_params": {
        "convex": false, // convex case not covered in the paper, but still available
        "groupsize": 4, // input dimension of the potentials, d in the paper
        "lamb_init": 5.0, // log of the lambda parameter
        "scaling": 0.01 // control the magnitude of the mu function
    },
    "multi_convolution": { 
        "num_channels": [
            1,
            4,
            8,
            60 // corresponds to (K times d) in the paper
        ],
        "size_kernels": [
            3,
            5,
            5
        ]
    },
    "noise_range": [
        0,
        0.2
    ],
    "noise_val": [ // corresponds to the value of the noise time 255, as often reported
        5,
        15,
        25,
        50
    ],
    "optimization": {
        "fixed_point_solver_bw_params": {
            "max_iter": 25,
            "tol": 0.001
        },
        "fixed_point_solver_fw_params": {
            "max_iter": 300,
            "tol": 0.0001
        }
    },
    "train_dataloader": {
        "batch_size": 128,
        "train_data_file": "path/to/train",
        "val_data_file": "path/to/val"
    },
    "training_options": {
        "lr_activation": 0.05,
        "lr_conv": 0.005,
        "lr_decay": 0.75,
        "lr_mu": 0.005,
        "n_batch_decay": 500,
        "n_batches": 5000
    }
}
```
