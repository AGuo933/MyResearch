import torch
import torch.nn as nn
from models.encoder import Encoder, EncoderLayer
from models.attn import ProbAttention, AttentionLayer
from models.embed import DataEmbedding


class Informer(nn.Module):
    def __init__(
        self,
        enc_in=16,
        d_model=128,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.2,
    ):
        super(Informer, self).__init__()

        # CNN feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Encoding
        self.enc_embedding = DataEmbedding(
            64, d_model, embed_type="timeF", freq="h", dropout=dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False,
                            factor=5,
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
                    activation="gelu",
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        # Output projection
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, time_features):
        batch_size = x.size(0)

        # 调整输入维度
        x = x.squeeze(1)  # [batch_size, feature_dim]
        x = x.unsqueeze(2)  # [batch_size, feature_dim, 1]

        # CNN feature extraction
        x = self.conv1(x)  # [batch_size, 32, 1]
        x = self.conv2(x)  # [batch_size, 64, 1]

        x = x.transpose(1, 2)  # [batch_size, 1, 64]

        # Transformer processing
        enc_out = self.enc_embedding(x, time_features)  # [batch_size, 1, d_model]
        enc_out, _ = self.encoder(enc_out)

        # 使用最后一个时间步的特征
        enc_out = enc_out[:, -1, :]  # [batch_size, d_model]
        output = self.fc(enc_out)  # [batch_size, 1]

        return output
