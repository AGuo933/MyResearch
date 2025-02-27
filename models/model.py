import torch
import torch.nn as nn
from models.encoder import Encoder, EncoderLayer
from models.attn import ProbAttention, AttentionLayer
from models.embed import DataEmbedding
import torch.nn.functional as F
import logging


class Informer(nn.Module):
    def __init__(
        self,
        enc_in=16,
        d_model=128,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.1,
    ):
        super(Informer, self).__init__()

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
                            factor=5,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="gelu",
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        # 预测头
        self.pre_fc = nn.Linear(d_model, 32)
        # 修改全连接层的输入维度为 64 * 128 = 8192
        self.fc = nn.Linear(64 * d_model, 1)  # 64是seq_len

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, time_features):
        try:
            # Input: x: [batch_size, feature_dim] = [64, 16]
            #        time_features: [batch_size, 3] = [64, 3]
            
            # 添加seq_len维度
            x = x.unsqueeze(1)  # [64, 1, 16]
            
            # Transformer processing
            enc_out = self.enc_embedding(x, time_features)  # [64, 64, 128]
            
            # Encoder layers处理
            enc_out, _ = self.encoder(enc_out)  # [64, 64, 128]

            # 展平所有时间步的特征
            enc_out = enc_out.reshape(enc_out.size(0), -1)  # [64, 8192]
            
            # 多层预测头
            enc_out = self.fc(enc_out)  # [64, 1]

            return enc_out  # [64, 1]
        except Exception as e:
            logging.error(f"Error in model forward pass: {str(e)}")
            logging.error(
                f"Input shapes - x: {x.shape}, time_features: {time_features.shape}"
            )
            return None
