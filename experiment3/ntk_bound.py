import torch
import traceback

import torch.nn as nn
import polars as pl
import numpy as np

from tqdm import tqdm
from time import time
from torch.utils.data import TensorDataset, DataLoader
from functorch import make_functional, vmap, vjp, jvp, jacrev


# Global variables
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Jindong, Wang (2020) Transfer Learning library[Source code]. 
# https://github.com/jindongwang/transferlearning/blob/cb9dac932fed5ad38156df4b6234b1e13325b9e5/code/distance/mmd_pytorch.py#L45
class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


def empirical_ntk_exact(fmodel_single, params, x1, x2):
    """
    Compute the exact empirical ntk matrix (no approximations). 
    The exact computation is memory-intensive, as it requires
    storing the gradients of all parameters.

    Parameters
    ----------
    fmodel_single : FunctionalModule
        functional model that accepts a single input tensor
    params : tensor
        tensor of network parameters (should be parameters of fmodel_single)
    x1 : tensor
        the first model input batch
    x2: tensor
        the second model input batch

    Returns
    -------
    tensor
        NTK matrix
    """
    # Compute vector gradient(x1)
    gradients1 = vmap(jacrev(fmodel_single), (None, 0))(params, x1)
    gradients1 = [g.flatten(2).double() for g in gradients1]
    
    # Compute vector gradient(x2)
    gradients2 = vmap(jacrev(fmodel_single), (None, 0))(params, x2)
    gradients2 = [g.flatten(2).double() for g in gradients2]
    
    # Compute J(x1) @ J(x2).T
    ntk = torch.stack([torch.einsum('Naf,Mbf->NMab', g1, g2) for g1, g2 in zip(gradients1, gradients2)])
    ntk = ntk.sum(0)
    return ntk


def empirical_ntk_batched(model, inputs, n_outputs=None, batch_size=1024, device='cuda'):
    """Compute the empirical neural tangent kernel of a PyTorch model on given inputs.
    Uses the recommended guidelines of computing gradients of model in PyTorch.
    Args:
        model: PyTorch model for which to compute the NTK.
        inputs: PyTorch tensor of inputs (shape: N x D).
        n_outputs (optional): Number of output dimensions of the model. If None, will be inferred
            from a forward pass through the model with one input.
        batch_size (optional): Batch size to use for computing the Jacobian in batches. Defaults to 1024.
        device (optional): Device to use for computations. Defaults to 'cuda'.

    Returns:
        A PyTorch tensor containing the empirical NTK (shape: N x N).

    """
    # Set the device for computations
    model.to(device)
    inputs = inputs.to(device)

    # Set the number of output dimensions if not provided
    if n_outputs is None:
        with torch.no_grad():
            n_outputs = model(inputs[:1]).shape[1]

    # Compute the Jacobian in batches
    n_inputs = inputs.shape[1]
    n_samples = inputs.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    jac_list = []
    for i in range(n_batches):
        inputs_batch = inputs[i*batch_size:(i+1)*batch_size]
        with torch.enable_grad():
            jac_batch = torch.func.jacrev(model, inputs_batch, create_graph=False)
        jac_batch = jac_batch.reshape(inputs_batch.shape[0], n_outputs, n_inputs)
        jac_batch = jac_batch.view(inputs_batch.shape[0], -1)
        jac_list.append(jac_batch)
    jac = torch.cat(jac_list, dim=0)

    # Compute the empirical NTK
    ntk = torch.mm(jac, jac.t())

    return ntk



def mmd_batch(X, Y, kernel_func, batch_size=100, device='cpu'):
    """
    Computes Maximum Mean Discrepancy (MMD) between two datasets X and Y using kernel_func and allows computation in batches.
    
    Args:
    - X: tensor of shape (n_samples_x, n_features) representing the first dataset
    - Y: tensor of shape (n_samples_y, n_features) representing the second dataset
    - kernel_func: function that takes two tensors of shape (n_samples, n_features) and returns a tensor of shape (n_samples, n_samples)
    - batch_size: batch size for computing MMD
    - device: device on which to perform computations
    
    Returns:
    - mmd: MMD between X and Y
    """
    
    n_samples_x = X.shape[0]
    n_samples_y = Y.shape[0]
    
    # compute number of batches
    n_batches_x = (n_samples_x - 1) // batch_size + 1
    n_batches_y = (n_samples_y - 1) // batch_size + 1
    
    # initialize variables for accumulating sums
    K_xx_sum = torch.zeros(1, device=device)
    K_yy_sum = torch.zeros(1, device=device)
    K_xy_sum = torch.zeros(1, device=device)
    
    # iterate over batches of X
    for i in range(n_batches_x):
        # get current batch of X
        start = i * batch_size
        end = min((i + 1) * batch_size, n_samples_x)
        X_batch = X[start:end].to(device)

        # compute kernel matrix for current batch of X
        K_xx_batch = kernel_func(X_batch, X_batch)

        # accumulate sum
        K_xx_sum += K_xx_batch.sum()

        # iterate over batches of Y
        for j in range(n_batches_y):
            # get current batch of Y
            start = j * batch_size
            end = min((j + 1) * batch_size, n_samples_y)
            Y_batch = Y[start:end].to(device)

            # compute kernel matrix for current batch of Y
            K_yy_batch = kernel_func(Y_batch, Y_batch)

            # accumulate sum
            K_yy_sum += K_yy_batch.sum()

            # compute kernel matrix between current batch of X and current batch of Y
            K_xy_batch = kernel_func(X_batch, Y_batch)

            # accumulate sum
            K_xy_sum += K_xy_batch.sum()

    # compute MMD
    mmd = (K_xx_sum / (n_samples_x ** 2)) + (K_yy_sum / (n_samples_y ** 2)) - (2 * K_xy_sum / (n_samples_x * n_samples_y))

    return mmd


def davinz_bound(
        model: nn.Module,
        predict,
        X_source: torch.Tensor, 
        X_target: torch.Tensor, 
        y_source: torch.Tensor
) -> float:
    """
    Compute the score of the valuation function as in DAVinz paper.

    Parameters
    ----------
    model : Module
        functional model that accepts a single input tensor
    X_source : tensor
        tensor from source feature dataset
    X_target : tensor
        tensor from target feature dataset
    y_source: tensor
        tensor from target variable dataset

    Returns
    -------
    float
        DAVinz valuation score, based on generaliztion bound
    """
    # Make function model
    fmodel, params = make_functional(model)
    fmodel.eval()
    fmodel_single = lambda params, x: fmodel(params, x.unsqueeze(0)).squeeze(0)
    
    # Compute NTK inverses
    X_source = X_source.to(DEVICE)
    ntk_mat = empirical_ntk_exact(fmodel_single, params, X_source, X_source).squeeze()
    # TODO: apply Gaussian noise to ensure invertibility
    ntk_mat = ntk_mat.cpu().double()
    for i in range(ntk_mat.shape[0]):
        if ntk_mat[i][i] == 0:
            ntk_mat[i][i] = 1e-16
    ntk_mat_inv = torch.inverse(ntk_mat.double()).double()
    
    # Compute MMDs
    # X_target = X_target.to(DEVICE)
    mmd_val = MMD_loss().forward(X_source.cpu().double(), X_target.cpu().double()).item()

    # Compute yhat estimate (yhat = y - f(x))
    yhat = (y_source.cpu().double() - predict(model, X_source).cpu().double()).squeeze().double()

    # Compute valuation score (bound)
    q = torch.matmul(torch.matmul(torch.t(yhat), ntk_mat_inv.cpu().double()), yhat)
    m = X_source.shape[0]
    sqrt_q = torch.sqrt(torch.div(q, m)).item()

    # Compute k
    k = 2
    
    return k, sqrt_q, mmd_val

