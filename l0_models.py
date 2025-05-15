import torch
import torch.nn as nn
from l0_layers import L0Conv2d, L0Dense
from copy import deepcopy
import torch.nn.functional as F
import numpy as np

def get_flat_fts(in_size, fts):
    dummy_input = torch.ones(1, *in_size)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    f = fts(torch.autograd.Variable(dummy_input))
    print('conv_out_size: {}'.format(f.size()))
    return int(np.prod(f.size()[1:]))

class L0PolicyNet(nn.Module):
    def __init__(self, num_classes=6, input_size = (1,35,35), input_dims=(1, 32, 32), conv_dims=(32, 32, 16), fc_dims=512,
                 N=50000, beta_ema=0., weight_decay=0.0, lambas=(0.01, 0.01, 0.01, 0.01), local_rep=False,
                 temperature=2./3.):
        super(L0PolicyNet, self).__init__()
        self.N = N
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.beta_ema = beta_ema
        self.weight_decay = weight_decay

        #convs = [L0Conv2d(input_dims[0], conv_dims[0], 3, stride=1, droprate_init=0.001, temperature=temperature,
        #                  weight_decay=self.weight_decay, lamba=lambas[0], local_rep=local_rep),
        #         nn.ReLU(),
        #         L0Conv2d(input_dims[1], conv_dims[1], 4, stride=2, droprate_init=0.001, temperature=temperature,
        #                  weight_decay=self.weight_decay, lamba=lambas[1], local_rep=local_rep),
        #         nn.ReLU(),
        #         L0Conv2d(input_dims[2], conv_dims[2], 4, stride=2, droprate_init=0.001, temperature=temperature,
        #                  weight_decay=self.weight_decay, lamba=lambas[1], local_rep=local_rep),
        #         nn.ReLU()]
        convs = [nn.Conv2d(1, 32, 3, stride=1), nn.ReLU(),
                nn.Conv2d(32, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 16, 4, stride=2), nn.ReLU(),]
        self.convs = nn.Sequential(*convs)
        if torch.cuda.is_available():
            self.convs = self.convs.cuda()

        flat_fts = get_flat_fts(input_size, self.convs) + 4

        fcs = [L0Dense(flat_fts, self.fc_dims, droprate_init=0.001, weight_decay=self.weight_decay,
                       lamba=lambas[2], local_rep=local_rep, temperature=temperature), nn.ReLU(),
               L0Dense(self.fc_dims, num_classes, droprate_init=0.001, weight_decay=self.weight_decay,
                       lamba=lambas[3], local_rep=local_rep, temperature=temperature)]
        #fcs = [nn.Linear(flat_fts, 512), nn.ReLU(), nn.Linear(512, num_classes)]
        self.fcs = nn.Sequential(*fcs)
        if torch.cuda.is_available():
            self.fcs = self.fcs.cuda()

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense) or isinstance(m, L0Conv2d):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, pixels, vel, angle):
        o = self.convs(pixels)
        o = o.view(o.size(0), -1)
        o = torch.cat([o, vel, angle], dim=1)
        return self.fcs(o)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params