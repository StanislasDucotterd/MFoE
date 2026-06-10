import torch


def HBR(x_init, y, Hty, lip, model, sigma, max_iter=300, tol=1e-4):

    # initial value: noisy image
    x = torch.clone(x_init)
    x_old = torch.clone(x_init)

    # cache values of scaling coeff for efficiency
    scaling = model.get_scaling(sigma=sigma)

    # the index of the images that have not converged yet
    idx = torch.arange(0, x.shape[0], device=x.device)
    res = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

    grad, cost = model.reconstruct(x, y, sigma=sigma)

    alpha = 1.99 / lip
    beta = 0.5 * torch.ones(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)

    # mean number of iterations over the batch
    i_mean = 0
    for i in range(max_iter):
        model.scaling = scaling[idx]
        z = x[idx] - alpha * grad[idx] + beta[idx] * (x[idx] - x_old[idx])
        beta = 0.5 * (beta + 1.)
        new_grad, new_cost = model.reconstruct(z, y[idx], Hty[idx], sigma=sigma[idx])
        decrease = alpha * (1 - lip * alpha / 2) * grad[idx].pow(2).sum(dim=(1, 2, 3))
        restart = (new_cost > cost[idx] - decrease)

        x_old = x.clone()

        x[idx[~restart]] = z[~restart]
        grad[idx[~restart]] = new_grad[~restart]
        cost[idx[~restart]] = new_cost[~restart]

        if restart.any():
            model.scaling = scaling[idx[restart]]
            beta[idx[restart]] = 0.5
            x[idx[restart]] = x[idx[restart]] - alpha * grad[idx[restart]]
            grad[idx[restart]], cost[idx[restart]] = model.reconstruct(x[idx[restart]], y[idx[restart]], sigma=sigma[idx[restart]])

        if i > 0:
            num = torch.linalg.vector_norm(x[idx] - x_old[idx], dim=(1, 2, 3))
            den = torch.linalg.vector_norm(x[idx], dim=(1, 2, 3))
            res = num / (den + 1e-8)

        idx = idx[res > tol]
        i_mean += torch.sum(res > tol).item() / x.shape[0]
        
        if len(idx) == 0:
            break

    model.clear_cache()

    return x, i+1, i_mean+1

def proj_l1_channel(x):
    """
    Projects a batch of vectors x in dimension d onto the unit l1-ball.
    The dimensions d and K are defined in the paper, H and W are the height and width of the image.

    Args:
        x torch.Tensor: Input tensor of shape (batch, d, K, H, W).
    """
    norm_x = torch.norm(x, p=1, dim=1, keepdim=True)
    mask = (norm_x <= 1.0).repeat(1, x.shape[1], 1, 1, 1)

    if mask.all():
        return x  # Already within the l1-ball

    abs_x = torch.abs(x)
    sorted_x, _ = torch.sort(abs_x, descending=True, dim=1)
    cumsum_x = torch.cumsum(sorted_x, dim=1)

    input_dim = x.shape[1]
    rho = (sorted_x * torch.arange(1, input_dim + 1, device=x.device)
           [None, :, None, None, None] > (cumsum_x - 1.0)).sum(dim=1) - 1

    # This is necessary due to numerical errors
    rho[rho < 0] = 0

    theta = (torch.gather(cumsum_x, 1, rho.unsqueeze(1)) - 1.0) / \
        (rho + 1).unsqueeze(1)
    projected_x = torch.sign(x) * torch.clamp(abs_x - theta, min=0)
    projected_x[mask] = x[mask]

    return projected_x
