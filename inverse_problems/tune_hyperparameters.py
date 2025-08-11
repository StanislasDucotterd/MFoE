import torch


def tune_hyperparameters(test, lamb_init, sigma_init):
    """
    Function to find the best hyperparameter combination through zeroth order optimization.
    """

    best_lamb = lamb_init
    best_sigma = sigma_init
    gamma = torch.tensor(4.0)
    psnrs = {}
    tests = 0

    best_psnr = 0.
    while gamma > 1.05:
        params1 = [best_lamb/gamma, best_lamb, best_lamb*gamma]
        params2 = [best_sigma/gamma, best_sigma, best_sigma*gamma]
        for lamb in params1:
            for sigma in params2:
                if len(psnrs) > 0:
                    prev_hyper = torch.tensor([*psnrs.keys()])
                    prev_hyper[:,0] = (prev_hyper[:,0] - lamb).abs() / lamb
                    prev_hyper[:,1] = (prev_hyper[:,1] - sigma).abs() / sigma
                    if prev_hyper.sum(dim=1).min() < 1e-2:
                        continue
                tests += 1
                mean_psnr = test(lamb, sigma)
                print(f'Lambda: {lamb:.6}, Sigma: {sigma:.6}, PSNR: {mean_psnr:.4f}')
                psnrs[(lamb, sigma)] = mean_psnr
        if max(psnrs.values()) > best_psnr + 1e-3:
            best_psnr = max(psnrs.values())
            best_lamb, best_sigma = max(psnrs, key=psnrs.get)
        else:
            gamma = torch.sqrt(gamma)
            if gamma.item() > 1.05: print('New gamma:', gamma.item())
    return best_lamb, best_sigma