import torch.nn as nn


def as_list(value, length):
    return list(value) if isinstance(value, (list, tuple)) else [value] * length


class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes=10, dropout=0.0, batch_norm=False, bn_momentum=0.1):
        super().__init__()
        count = len(hidden_sizes)
        layers = [nn.Flatten()]
        size = input_size
        for hidden, p, use_bn in zip(hidden_sizes, as_list(dropout, count), as_list(batch_norm, count)):
            layers.append(nn.Linear(size, hidden))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden, momentum=bn_momentum))
            layers.append(nn.ReLU())
            if p > 0:
                layers.append(nn.Dropout(p))
            size = hidden
        layers.append(nn.Linear(size, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_dropout(model, p):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = p


def layer_weights(model):
    return [m.weight.detach().cpu().flatten() for m in model.modules() if isinstance(m, nn.Linear)]
