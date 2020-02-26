import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal
from Learning.Networks.network_modules_torch import *

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(ValueNetwork, self).__init__()

        ff1_meta = {
            "activation": None, "input_size": input_dim,
            "output_size": 64, "initialization": "orthogonal"}
        self.ff1 = NetworkConnectivityModule("linear", ff1_meta)
        ff2_meta = {
            "activation": None, "input_size": 64,
            "output_size": 64, "initialization": "orthogonal"}
        self.ff2 = NetworkConnectivityModule("linear", ff2_meta)
        ff3_meta = {
            "activation": None, "input_size": 64,
            "output_size": output_dim, "initialization": "orthogonal"}
        self.ff3 = NetworkConnectivityModule("linear", ff3_meta)

    def forward(self, x):
        x = torch.relu(self.ff1(x))
        x = torch.relu(self.ff2(x))
        x = self.ff3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, output_dim=1):
        super(QNetwork, self).__init__()

        # Q1 architecture
        ff1_1_meta = {
            "activation": None, "input_size": num_inputs + num_actions,
            "output_size": 256, "initialization": "orthogonal"}
        self.ff1_1 = NetworkConnectivityModule("linear", ff1_1_meta)
        ff1_2_meta = {
            "activation": None, "input_size": 256,
            "output_size": 256, "initialization": "orthogonal"}
        self.ff1_2 = NetworkConnectivityModule("linear", ff1_2_meta)
        ff1_3_meta = {
            "activation": None, "input_size": 256,
            "output_size": output_dim, "initialization": "orthogonal"}
        self.ff1_3 = NetworkConnectivityModule("linear", ff1_3_meta)

        # Q2 architecture
        ff2_1_meta = {
            "activation": None, "input_size": num_inputs + num_actions,
            "output_size": 256, "initialization": "orthogonal"}
        self.ff2_1 = NetworkConnectivityModule("linear", ff2_1_meta)
        ff2_2_meta = {
            "activation": None, "input_size": 256,
            "output_size": 256, "initialization": "orthogonal"}
        self.ff2_2 = NetworkConnectivityModule("linear", ff2_2_meta)
        ff2_3_meta = {
            "activation": None, "input_size": 256,
            "output_size": output_dim, "initialization": "orthogonal"}
        self.ff2_3 = NetworkConnectivityModule("linear", ff2_3_meta)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = torch.relu(self.ff1_1(xu))
        x1 = torch.relu(self.ff1_2(x1))
        x1 = self.ff1_3(x1)

        x2 = torch.relu(self.ff2_1(xu))
        x2 = torch.relu(self.ff2_2(x2))
        x2 = self.ff2_3(x2)

        return x1, x2


class PongNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, action_high, action_low):
        super(PongNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.action_bias = torch.FloatTensor((action_high + action_low) / 2.)
        self.action_scale = torch.FloatTensor((action_high - action_low) / 2.)

        ff1_meta = {
            "activation": None, "input_size": input_dim,
            "output_size": 256, "initialization": "orthogonal"}
        self.ff1 = NetworkConnectivityModule("linear", ff1_meta)

        ff2_meta = {
            "activation": None, "input_size": 256,
            "output_size": 256, "initialization": "orthogonal"}
        self.ff2 = NetworkConnectivityModule("linear", ff2_meta)

        ff3_meta = {
            "activation": None, "input_size": 256,
            "output_size": output_dim, "initialization": "orthogonal"}
        self.ff3 = NetworkConnectivityModule("linear", ff3_meta)

        log_std_linear_meta = {
            "activation": None, "input_size": 256,
            "output_size": output_dim, "initialization": "orthogonal"}
        self.log_std_linear = NetworkConnectivityModule("linear", log_std_linear_meta)

    def forward(self, x):
        x = torch.relu(self.ff1(x))
        x = torch.relu(self.ff2(x))
        mean = self.ff3(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean




















