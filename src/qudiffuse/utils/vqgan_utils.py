import copy
import torch
import torch.nn.functional as F

def normalize(in_channels):
    # Choose appropriate number of groups that divides into in_channels
    # Common choices: 32, 16, 8, 4, 2, 1
    num_groups = min(32, in_channels)
    
    # Ensure num_groups divides evenly into in_channels
    while in_channels % num_groups != 0:
        num_groups = num_groups // 2
        if num_groups < 1:
            num_groups = 1
            break
    
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


@torch.jit.script
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def calculate_adaptive_weight(recon_loss, g_loss, last_layer, disc_weight_max):
    recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
    return d_weight


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)