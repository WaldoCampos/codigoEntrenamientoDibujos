import copy
import math
import random
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class CosineDecayEMA():
    def __init__(self, tau, max_steps):
        super().__init__()
        self.base_tau = tau
        self.curr_step = 0
        self.max_steps = max_steps

    def update_average(self, old, new):
        if old is None:
            return new
        tau = 1 - (1-self.base_tau)*(math.cos(math.pi*self.curr_step/self.max_steps)+1)/2
        self.curr_step += 1
        return old * tau + (1 - tau) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )
    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer
        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)
        return representation

# main class

class BYOL2(nn.Module):
    def __init__(
            self,
            net,
            image_size,
            hidden_layer = -2,
            projection_size = 256,
            projection_hidden_size = 4096,
            augment_fn = None,
            augment_fn2 = None,
            moving_average_decay = 0.99,
            use_momentum = True,
            cosine_ema_steps = None
    ):
        super().__init__()
        self.net = net
        device = get_module_device(net)

        self.online_encoder = NetWrapper(net, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_projector = None
        if cosine_ema_steps:
            self.target_ema_updater = CosineDecayEMA(moving_average_decay, cosine_ema_steps)
        else:
            self.target_ema_updater = EMA(moving_average_decay)

        dummy = self.online_encoder(torch.randn(2, 3, image_size, image_size, device=device))
        self.online_projector = MLP(dummy.shape[1], projection_size, projection_hidden_size)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        
        self.to(device)

        self.augment1 = augment_fn
        self.augment2 = augment_fn2 if augment_fn2 else augment_fn

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder
    
    @singleton('target_projector')
    def _get_target_projector(self):
        target_projector = copy.deepcopy(self.online_projector)
        set_requires_grad(target_projector, False)
        return target_projector

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
        update_moving_average(self.target_ema_updater, self.target_projector, self.online_projector)

    def forward(
            self,
            x,
            return_embedding = False,
            return_projection = True
    ):

        if return_embedding == 'online':
            return self.online_encoder(x, return_projection = return_projection)
        elif return_embedding == 'target':
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            return target_encoder(x, return_projection = return_projection)

        image_one, image_two = self.augment1(x), self.augment2(x)

        online_embed_one = self.online_encoder(image_one)
        online_proj_one = self.online_projector(online_embed_one)
        online_pred_one = self.online_predictor(online_proj_one)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_projector = self._get_target_projector() if self.use_momentum else self.online_projector
            target_embed_two = target_encoder(image_two)
            target_proj_two = target_projector(target_embed_two)
            target_proj_two.detach_()

        loss = loss_fn(online_pred_one, target_proj_two.detach())
        return loss.mean()
