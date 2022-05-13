import torch
import torch.nn as nn
from torch import autograd
from Configs import configs

def calc_Dw_loss(probs : torch.Tensor, labels : int):
    """
    Calculate the loss for the discriminator."""
    labels = torch.full((probs.size(0),), labels, dtype=torch.float,device = configs.device)
    criterion = nn.BCELoss()

    adverloss = criterion(probs, labels)
    return adverloss

def R1_regularization(r1_coeff,probs,ws):
    return (r1_coeff/2)*compute_grad2(probs,ws).mean()

def compute_grad2(probs,w_input):
    batch_size = w_input.size(0)
    grad_dout = autograd.grad(
        outputs=probs.sum(),
        inputs=w_input,
        create_graph= True,
        retain_graph=True,
        only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    return grad_dout2.view(batch_size, -1).sum(1)