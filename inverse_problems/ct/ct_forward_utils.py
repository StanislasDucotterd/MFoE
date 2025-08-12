import torch_wrapper
import odl
import os
import sys
import torch
import numpy as np
sys.path.append(os.path.dirname(__file__))


def get_operators(img_size, space_range, num_angles, det_shape, device=None, fix_scaling=True):

    space = odl.uniform_discr([-space_range, -space_range], [space_range, space_range],
                              (img_size, img_size), dtype='float32', weighting=None)
    geometry = odl.tomo.geometry.parallel.parallel_beam_geometry(
        space, num_angles=num_angles, det_shape=det_shape)

    fwd_op_odl = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

    fbp_op_odl = odl.tomo.fbp_op(fwd_op_odl)

    adjoint_op_odl = fwd_op_odl.adjoint

    # fix scaling issues of the adjoint
    if fix_scaling:
        # we check the scaling on a circular mask
        fbp_op_odl = fbp_op_odl * \
            get_fbp_scaling(img_size, fwd_op_odl, fbp_op_odl)

        # we check the scaling by comparing <y, Hx>  and <H^T y, x> for a random input
        scaling = get_adjoint_scaling(
            img_size, num_angles, det_shape, fwd_op_odl, adjoint_op_odl)
        scaled_fwd_op_odl = fwd_op_odl * scaling
        adjoint_op_odl = scaled_fwd_op_odl.adjoint

    # the constant 90.65 is added because I have a different version
    # of the odl library installed, which has a different scaling
    fwd_op = torch_wrapper.OperatorModule(fwd_op_odl * 90.65).to(device)
    fbp_op = torch_wrapper.OperatorModule(fbp_op_odl / 90.65).to(device)
    adjoint_op = torch_wrapper.OperatorModule(
        adjoint_op_odl * 90.65).to(device)

    return fwd_op, fbp_op, adjoint_op


def get_adjoint_scaling(img_size, num_angles, det_shape, fwp, adjoint, inverse=True):
    x0 = np.random.rand(img_size, img_size)
    y0 = np.random.rand(num_angles, det_shape)

    Hx0 = fwp(x0)
    s1 = np.sum(y0*Hx0)
    s2 = np.sum(adjoint(y0)*x0)

    return (s1/s2)


def get_fbp_scaling(img_size, fwd, fbp):

    x0 = create_circular_mask(img_size)
    sinogram = fwd(x0)
    y = fbp(sinogram)

    # check in the center
    y_center = y * x0

    return x0.mean()/np.mean(y_center).item()


def create_circular_mask(img_size):
    center = (int(img_size/2), int(img_size/2))

    Y, X = np.ogrid[:img_size, :img_size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= img_size//2

    return mask


def get_op_norm(img_size, fwd_op, adjoint_op, device, n_iter=50):
    x = torch.rand((1, 1, img_size, img_size)).to(device)

    for i in range(n_iter):
        x = x / x.norm()
        x = adjoint_op(fwd_op(x))

    return x.norm().sqrt().item()
