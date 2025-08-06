import torch
import torch.nn as nn
import numpy as np
import copy

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=None, bias=True):
        super().__init__()
        if rank is None:
            rank = min(in_features, out_features) // 4
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.U = nn.Parameter(torch.Tensor(in_features, rank))
        self.V = nn.Parameter(torch.Tensor(rank, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.U, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.U)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = input @ self.U @ self.V
        if self.bias is not None:
            output += self.bias
        return output

    @staticmethod
    def from_linear(linear_layer, rank=None):
        W = linear_layer.weight.data.t()
        in_f, out_f = W.shape
        if rank is None:
            rank = min(in_f, out_f) // 4
        rank = min(rank, in_f, out_f)
        low_rank = LowRankLinear(in_f, out_f, rank, bias=(linear_layer.bias is not None))
        U_full, S, V_full = torch.svd(W)
        low_rank.U.data = U_full[:, :rank] * S[:rank].unsqueeze(0)
        low_rank.V.data = V_full.t()[:rank, :]
        if linear_layer.bias is not None:
            low_rank.bias.data = linear_layer.bias.data.clone()
        return low_rank

def create_low_rank_model(model, rank_factor=4):
    new_model = copy.deepcopy(model)
    skip = ['qkv', 'norm', 'cls_token', 'pos_embed']
    replaced = skipped = 0
    modules = dict(new_model.named_modules())
    for name, module in modules.items():
        if not isinstance(module, nn.Linear):
            continue
        if any(k in name for k in skip):
            skipped += 1
            continue
        rank = min(module.in_features, module.out_features) // rank_factor
        if rank < 2:
            skipped += 1
            continue
        low_rank = LowRankLinear.from_linear(module, rank=rank)
        parent_name, attr = (name.rsplit('.', 1) + [''])[:2]
        parent = modules[parent_name] if parent_name else new_model
        setattr(parent, attr, low_rank)
        replaced += 1
    print(f"Replaced {replaced} layers, skipped {skipped}")
    return new_model

