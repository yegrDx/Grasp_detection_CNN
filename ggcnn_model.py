# ggcnn_model.py
# GG-CNN: архитектура как в оригинальном репозитории (3 Conv + 3 ConvTranspose).
# Возвращаем dict, совместимый с train.py; лосс — MSE, как в оригинале.

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]

class GGCNN(nn.Module):
    """
    GG-CNN (эквивалент Keras-версии из RSS-пейпера)
    """
    def __init__(self, input_channels: int = 1):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(input_channels,   filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
        self.conv2 = nn.Conv2d(filter_sizes[0],  filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
        self.conv3 = nn.Conv2d(filter_sizes[1],  filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)
        # Decoder (ConvTranspose2d параметры — как в оригинале)
        self.convt1 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=3, output_padding=1)

        # Heads (kernel_size=2 как в оригинале)
        self.pos_output   = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.cos_output   = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.sin_output   = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)

        # Xavier init — как в оригинале
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        pos   = self.pos_output(x)
        cos   = self.cos_output(x)
        sin   = self.sin_output(x)
        width = self.width_output(x)

        # В оригинале сигмоиду на pos можно не ставить — оставим как есть для совместимости с MSE
        return {'pos': pos, 'cos': cos, 'sin': sin, 'width': width}

def ggcnn_loss(pred: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # MSE-лоссы как в оригинале
    p_loss     = F.mse_loss(pred['pos'],   y['pos'])
    cos_loss   = F.mse_loss(pred['cos'],   y['cos'])
    sin_loss   = F.mse_loss(pred['sin'],   y['sin'])
    width_loss = F.mse_loss(pred['width'], y['width'])
    total = p_loss + cos_loss + sin_loss + width_loss
    return {
        'loss': total,
        'p_loss': p_loss.detach(),
        'cos_loss': cos_loss.detach(),
        'sin_loss': sin_loss.detach(),
        'width_loss': width_loss.detach()
    }
