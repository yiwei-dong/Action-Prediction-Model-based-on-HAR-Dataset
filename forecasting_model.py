import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, dimension_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, dimension_model, 2) * (-(math.log(10000.0) / dimension_model)))
        positional_encoding = torch.zeros(max_len, dimension_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        # x: [batch, seq_len, dimension_model]
        x = x + self.positional_encoding[:x.size(1), :]
        return x

class HARTransformer(nn.Module):
    def __init__(self, input_dimension=561, num_classes=6, dimension_model=128, nhead=4,
                 num_layers=3, dimension_feedforward=256, dropout=0.1):
        super().__init__()
        # 输入嵌入层
        self.embedding = nn.Linear(input_dimension, dimension_model)
        self.positional_encoding = PositionalEncoding(dimension_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dimension_model,
            nhead=nhead,
            dim_feedforward=dimension_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类
        self.classifier = nn.Sequential(
            nn.Linear(dimension_model, dimension_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dimension_model // 2, num_classes)
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for parameter in self.parameters():
            # 遍历模型的所有可训练参数。
            # self.parameters() 是 PyTorch 中的一个方法，返回模型中所有可训练的参数（张量）
            if parameter.dim() > 1:
                # 如果维度大于1，说明这是一个权重矩阵（如全连接层的权重），而不是偏置向量（偏置通常是1维的）
                nn.init.xavier_uniform_(parameter)
                # 使用 Xavier 均匀分布初始化方法对权重矩阵进行初始化。
                # Xavier 初始化的目的是确保每一层的输入和输出的方差相同，从而加速神经网络的训练。

    def forward(self, x):
        # x shape: [batch_size, input_dimension]
        # x = x.unsqueeze(1)  # 添加序列维度 [batch_size, 1, input_dimension]

        x = self.embedding(x)  # [batch_size, 1, dimension_model]
        x = self.positional_encoding(x)

        x = self.transformer_encoder(x)  # [batch_size, 1, dimension_model]
        x = x.squeeze(1)  # [batch_size, dimension_model]
        x = self.classifier(x)

        return x