import torch
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim


def orthogonal_init(tensor, gain=1.0):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, initialization_type, scale=2**0.5):
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")


class NetworkConnectivityModule(nn.Module):
    def __init__(self, module_type, module_metadata):
        """
        Network Sub-Module responsible for processing
         and connecting information between two layers
        :param module_type: (str) -> module connection type
        :param module_metadata: (dict) -> dictionary containing relavent metadata
        """
        super(NetworkConnectivityModule, self).__init__()
        self.module_type = module_type
        self.module_metadata = module_metadata
        self.activation = self.module_metadata["activation"]
        if "initialization" in self.module_metadata:
            self.initialization = self.module_metadata["initialization"]
        else:
            self.initialization = None

        # todo: small intialization, especially for neuromodulation
        # todo: grad clip? weight norm?
        if self.module_type == "linear":
            self.layer = nn.Linear(
                module_metadata["input_size"], module_metadata["output_size"])

            if self.initialization is not None:
                initialize_weights(self.layer, self.initialization)

        elif self.module_type == "eligibility":
            self.hebbian_trace = torch.zeros(
                (module_metadata["input_size"], module_metadata["output_size"]), requires_grad=False)
            self.eligibility_trace = torch.zeros(
                (module_metadata["input_size"], module_metadata["output_size"]), requires_grad=False)

            self.modulation_fan_in = nn.Linear(module_metadata["output_size"], 1)
            self.modulation_fan_out = nn.Linear(1, module_metadata["output_size"])
            self.eligibility_eta = nn.Parameter(tensor=torch.ones(1), requires_grad=True)
            self.alpha_plasticity = nn.Parameter(tensor=torch.ones(
                module_metadata["output_size"], module_metadata["output_size"]), requires_grad=True)

            self.layer = nn.Linear(
                module_metadata["input_size"], module_metadata["output_size"])

        elif self.module_type == "eligibility_recurrent":
            self.recurrent_trace = torch.zeros(
                module_metadata["output_size"], 1)
            self.hebbian_trace = torch.zeros(
                (module_metadata["input_size"], module_metadata["output_size"]), requires_grad=False)
            self.eligibility_trace = torch.zeros(
                (module_metadata["input_size"], module_metadata["output_size"]), requires_grad=False)

            self.modulation_fan_in = nn.Linear(module_metadata["output_size"], 1)
            self.modulation_fan_out = nn.Linear(1, module_metadata["output_size"])
            self.eligibility_eta = nn.Parameter(tensor=torch.ones(1), requires_grad=True)
            self.alpha_plasticity = nn.Parameter(tensor=torch.ones(), requires_grad=True)

            self.recurrent_layer = nn.Linear(
                module_metadata["output_size"], module_metadata["output_size"])

            self.layer = nn.Linear(
                module_metadata["input_size"], module_metadata["output_size"])

    def reset(self):
        if self.module_type == "eligibility":
            self.hebbian_trace = self.hebbian_trace.detach()*0
            self.eligibility_trace = self.hebbian_trace.detach()*0

        elif self.module_type == "eligibility_recurrent":
            pass

    def update_trace(self, pre_synaptic, post_synaptic):
        if self.module_type == "eligibility":
            modulatory_signal = self.modulation_fan_out(
                torch.tanh(self.modulation_fan_in(post_synaptic)))

            self.hebbian_trace = torch.clamp(
                self.hebbian_trace + modulatory_signal*self.eligibility_trace,
                max=self.module_metadata["clip"], min=self.module_metadata["clip"]*-1)

            self.eligibility_trace = (torch.ones(1)-self.eligibility_eta)*\
                self.eligibility_trace + self.eligibility_eta*(torch.mm(pre_synaptic, post_synaptic))

        elif self.module_type == "eligibility_recurrent":
            modulatory_signal = self.modulation_fan_out(
                torch.tanh(self.modulation_fan_in(post_synaptic)))

            self.hebbian_trace = torch.clamp(
                self.hebbian_trace + modulatory_signal * self.eligibility_trace,
                max=self.module_metadata["clip"], min=self.module_metadata["clip"] * -1)

            self.eligibility_trace = (torch.ones(1) - self.eligibility_eta)*\
                self.eligibility_trace + self.eligibility_eta*(torch.mm(pre_synaptic, post_synaptic))

    def forward(self, x):
            post_synaptic = None
            # is hebb detached...?
            pre_synaptic = x.clone()  # x.detach().clone()
            if self.module_type == "linear":
                post_synaptic = self.layer(x)
                if self.activation is not None:
                    post_synaptic = self.activation(post_synaptic)

            elif self.module_type == "eligibility":
                fixed_weights = self.layer(x)
                plastic_weights = x.mm(self.alpha_plasticity*self.hebbian_trace)
                post_synaptic = fixed_weights + plastic_weights
                if self.activation is not None:
                    post_synaptic = self.activation(post_synaptic)
                self.update_trace(pre_synaptic=pre_synaptic, post_synaptic=post_synaptic)

            elif self.module_type == "eligibility_recurrent":
                fixed_weights = self.layer(x)
                plastic_weights = x.mm(self.alpha_plasticity * self.hebbian_trace)
                post_synaptic = fixed_weights + plastic_weights
                if self.activation is not None:
                    post_synaptic = self.activation(post_synaptic)
                self.update_trace(pre_synaptic=pre_synaptic, post_synaptic=post_synaptic)

            return post_synaptic

















