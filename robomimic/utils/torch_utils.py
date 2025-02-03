"""
This file contains some PyTorch utilities.
"""
import numpy as np
import torch
import torch.optim as optim


def soft_update(source, target, tau):
    """
    Soft update from the parameters of a @source torch module to a @target torch module
    with strength @tau. The update follows target = target * (1 - tau) + source * tau.

    Args:
        source (torch.nn.Module): source network to push target network parameters towards
        target (torch.nn.Module): target network to update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.copy_(
            target_param * (1.0 - tau) + param * tau
        )


def hard_update(source, target):
    """
    Hard update @target parameters to match @source.

    Args:
        source (torch.nn.Module): source network to provide parameters
        target (torch.nn.Module): target network to update parameters for
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.copy_(param)


def get_torch_device(try_to_use_cuda):
    """
    Return torch device. If using cuda (GPU), will also set cudnn.benchmark to True
    to optimize CNNs.

    Args:
        try_to_use_cuda (bool): if True and cuda is available, will use GPU

    Returns:
        device (torch.Device): device to use for models
    """
    if try_to_use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def reparameterize(mu, logvar):
    """
    Reparameterize for the backpropagation of z instead of q.
    This makes it so that we can backpropagate through the sampling of z from
    our encoder when feeding the sampled variable to the decoder.

    (See "The reparameterization trick" section of https://arxiv.org/abs/1312.6114)

    Args:
        mu (torch.Tensor): batch of means from the encoder distribution
        logvar (torch.Tensor): batch of log variances from the encoder distribution

    Returns:
        z (torch.Tensor): batch of sampled latents from the encoder distribution that
            support backpropagation
    """
    # logvar = \log(\sigma^2) = 2 * \log(\sigma)
    # \sigma = \exp(0.5 * logvar)

    # clamped for numerical stability
    logstd = (0.5 * logvar).clamp(-4, 15)
    std = torch.exp(logstd)

    # Sample \epsilon from normal distribution
    # use std to create a new tensor, so we don't have to care
    # about running on GPU or not
    eps = std.new(std.size()).normal_()

    # Then multiply with the standard deviation and add the mean
    z = eps.mul(std).add_(mu)

    return z

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

def optimizer_from_optim_params(net_optim_params, net):
    """
    Helper function to return a torch Optimizer from the optim_params 
    section of the config for a particular network.

    Args:
        optim_params (Config): optim_params part of algo_config corresponding
            to @net. This determines the optimizer that is created.

        net (torch.nn.Module): module whose parameters this optimizer will be
            responsible

    Returns:
        optimizer (torch.optim.Optimizer): optimizer
    """
    if len(net_optim_params["optimizer"]) == 0:
        net_optim_params["optimizer"] = "Adam"
    if net_optim_params["optimizer"] == 'Adam':
        return optim.Adam(
            params=net.parameters(),
            lr=net_optim_params["learning_rate"]["initial"],
            weight_decay=net_optim_params["regularization"]["L2"],
        )
    elif net_optim_params["optimizer"] == 'SAM':
        base_optimizer = torch.optim.SGD
        return SAM(net.parameters(), base_optimizer, rho=2.0, adaptive=True, lr=net_optim_params["learning_rate"]["initial"],
                        momentum=0.9, weight_decay=net_optim_params["regularization"]["L2"])
    elif net_optim_params["optimizer"] == "SGD":
        return optim.SGD(
            params=net.parameters(),
            lr=net_optim_params["learning_rate"]["initial"],
            weight_decay=net_optim_params["regularization"]["L2"],
            momentum=0.9
        )
    else:
        raise NotImplementedError


def lr_scheduler_from_optim_params(net_optim_params, net, optimizer):
    """
    Helper function to return a LRScheduler from the optim_params 
    section of the config for a particular network. Returns None
    if a scheduler is not needed.

    Args:
        optim_params (Config): optim_params part of algo_config corresponding
            to @net. This determines whether a learning rate scheduler is created.

        net (torch.nn.Module): module whose parameters this optimizer will be
            responsible

        optimizer (torch.optim.Optimizer): optimizer for this net

    Returns:
        lr_scheduler (torch.optim.lr_scheduler or None): learning rate scheduler
    """
    lr_scheduler = None
    if len(net_optim_params["learning_rate"]["epoch_schedule"]) > 0:
        # decay LR according to the epoch schedule
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=net_optim_params["learning_rate"]["epoch_schedule"],
            gamma=net_optim_params["learning_rate"]["decay_factor"],
        )
    if net_optim_params["learning_rate"]["cosine_annealing"]:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=net_optim_params["learning_rate"]["t_max"], eta_min=1e-8)
    return lr_scheduler


def backprop_for_loss(net, optim, loss, max_grad_norm=None, retain_graph=False, closure=None):
    """
    Backpropagate loss and update parameters for network with
    name @name.

    Args:
        net (torch.nn.Module): network to update

        optim (torch.optim.Optimizer): optimizer to use

        loss (torch.Tensor): loss to use for backpropagation

        max_grad_norm (float): if provided, used to clip gradients

        retain_graph (bool): if True, graph is not freed after backward call

    Returns:
        grad_norms (float): average gradient norms from backpropagation
    """

    # backprop
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)

    # gradient clipping
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

    # compute grad norms
    grad_norms = 0.
    for p in net.parameters():
        # only clip gradients for parameters for which requires_grad is True
        if p.grad is not None:
            grad_norms += p.grad.data.norm(2).pow(2).item()

    # step
    if closure is not None:
        optim.step(closure)
    else:
        optim.step()

    return grad_norms


class dummy_context_mgr():
    """
    A dummy context manager - useful for having conditional scopes (such
    as @maybe_no_grad). Nothing happens in this scope.
    """
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False


def maybe_no_grad(no_grad):
    """
    Args:
        no_grad (bool): if True, the returned context will be torch.no_grad(), otherwise
            it will be a dummy context
    """
    return torch.no_grad() if no_grad else dummy_context_mgr()
