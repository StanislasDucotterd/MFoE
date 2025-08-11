# Multivariate Fields of Experts
Implementation of experiments done in : https://arxiv.org/abs/2508.06490v1

![alt text](https://github.com/StanislasDucotterd/MFoE/blob/main/potential.pdf?raw=true)

#### Description
We aim at the solution of inverse problems in imaging, by combining a penalized sparse representation of image patches with an unconstrained smooth one. This allows for a straightforward interpretation of the reconstruction. We formulate the optimization as a bilevel problem. The inner problem deploys classical algorithms while the outer problem optimizes the dictionary and the regularizer parameters through supervised learning. The process is carried out via implicit differentiation and gradient-based optimization. We evaluate our method for denoising, super-resolution, and compressed-sensing magnetic-resonance imaging. We compare it to other classical models as well as deep-learning-based methods and show that it always outperforms the former and also the latter in some instances. 

#### Requirements
The required packages:
- `pytorch`
- `torchdeq`
- `matplotlib`

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
        "convex": false, // the convex case is not covered in the paper, but still available
        "groupsize": 4, // corresponds to d in the paper
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
