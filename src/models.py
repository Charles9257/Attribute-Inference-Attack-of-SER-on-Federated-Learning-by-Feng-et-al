"""Models ported from the official implementation (usc-sail/fed-ser-leakage, model/*.py).

dnn_classifier : SER model (input -> 256 -> 128 -> 4), ReLU + dropout.
AttackModel    : CNN over the shared update of one layer (weight matrix as a 1-channel
                 image) concatenated with the bias update, -> 2-way gender prediction.
"""
import torch
import torch.nn as nn


class SERClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=4, dropout=0.2):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, 256)
        self.dense2 = nn.Linear(256, 128)
        self.pred_layer = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(0.2)  # fixed 0.2 in the original second block
        for m in [self.dense1, self.dense2, self.pred_layer]:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.dropout(torch.relu(self.dense1(x)))
        x = self.dropout2(torch.relu(self.dense2(x)))
        return torch.log_softmax(self.pred_layer(x), dim=1)


def _block(cin, cout, pool, pool_stride, p, bn=False):
    layers = [nn.Conv2d(cin, cout, kernel_size=5, padding=2)]
    if bn: layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.ReLU())
    if pool: layers.append(nn.MaxPool2d(kernel_size=pool, stride=pool_stride))
    layers.append(nn.Dropout2d(p))
    return layers


class AttackModel(nn.Module):
    """Attack CNN. Update tensors are laid out as in PyTorch (out_features x in_features),
    i.e. the first-layer weight update is 256 x input_dim."""

    def __init__(self, leak_layer, weight_shape, bias_dim, dropout=0.2):
        super().__init__()
        p = dropout
        if leak_layer == 'first':
            stride_len = 8  # emobase setting in the original (stride_len = 8)
            conv = _block(1, 16, 2, 2, p) + _block(16, 32, 4, 4, p) + _block(32, 64, stride_len, stride_len, p, bn=True)
        elif leak_layer == 'second':
            conv = _block(1, 16, 2, 2, p) + _block(16, 32, 4, 4, p) + _block(32, 64, 4, 4, p, bn=True)
        else:  # last
            conv = _block(1, 16, None, None, p) + _block(16, 32, 2, (1, 4), p) + _block(32, 64, 2, (1, 4), p, bn=True)
        self.conv = nn.Sequential(*conv)
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 1, *weight_shape)).numel()
        self.head = nn.Sequential(
            nn.Linear(n + bias_dim, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 2))
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, w, b):
        z = self.conv(w.unsqueeze(1)).flatten(1)
        return torch.log_softmax(self.head(torch.cat([z, b], 1)), dim=1)
