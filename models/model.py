import torch
import torch.nn as nn
from models.encoder import Encoder, EncoderLayer
from models.attn import ProbAttention, AttentionLayer
from models.embed import DataEmbedding
import torch.nn.functional as F


class Informer(nn.Module):
    def __init__(
        self,
        enc_in=16,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
    ):
        super(Informer, self).__init__()

        # 添加输入归一化
        self.input_norm = nn.LayerNorm(enc_in)

        # Encoding
        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed_type="timeF", freq="h", dropout=dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False,
                            factor=3,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="relu",
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        # Output projection with initialization
        self.pre_fc = nn.Linear(d_model, d_model // 2)
        self.fc = nn.Linear(d_model // 2, 1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, time_features):
        # 输入检查
        if torch.isnan(x).any():
            print("Input contains NaN!")
            return None

        # 输入归一化
        x = x.squeeze(1)  # [batch_size, feature_dim]
        x = self.input_norm(x)
        x = x.unsqueeze(1)  # [batch_size, 1, feature_dim]

        # Transformer processing
        enc_out = self.enc_embedding(x, time_features)

        # 移除残差连接，因为维度不匹配
        enc_out, _ = self.encoder(enc_out)

        # 使用最后一个时间步的特征
        enc_out = enc_out[:, -1, :]  # [batch_size, d_model]

        # 多层预测头
        enc_out = F.relu(self.pre_fc(enc_out))
        enc_out = self.fc(enc_out)

        return enc_out
