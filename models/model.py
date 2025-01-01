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
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
    ):
        super(Informer, self).__init__()

        # 添加输入归一化
        self.input_norm = nn.LayerNorm(enc_in)
        
        # 特征重要性权重
        self.feature_importance = nn.Parameter(torch.ones(enc_in))
        self.softmax = nn.Softmax(dim=0)
        
        # Encoding
        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed_type="timeF", freq="h", dropout=dropout
        )

        # 残差连接的投影层
        self.residual_proj = nn.Linear(enc_in, d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False,
                            factor=3,
                            attention_dropout=dropout,
                            output_attention=True,  # 启用注意力输出
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="gelu",  # 使用GELU激活函数
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

    def get_feature_importance(self):
        """返回特征重要性分数"""
        return self.softmax(self.feature_importance).detach().cpu().numpy()

    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, time_features):
        try:
            # 输入处理
            x = x.squeeze(1)
            x = self.input_norm(x)
            
            # 应用特征重要性权重
            feature_weights = self.softmax(self.feature_importance)
            x = x * feature_weights
            
            x = x.unsqueeze(1)

            # 创建残差连接
            residual = self.residual_proj(x.squeeze(1)).unsqueeze(1)

            # Transformer处理
            enc_out = self.enc_embedding(x, time_features)
            enc_out, attns = self.encoder(enc_out)

            # 添加残差连接
            enc_out = enc_out + residual
            
            # 使用最后一个时间步的特征
            enc_out = enc_out[:, -1, :]  # [batch_size, d_model]

            # 多层预测头
            enc_out = F.relu(self.pre_fc(enc_out))
            enc_out = self.fc(enc_out)

            return enc_out

        except Exception as e:
            logging.error(f"Error in model forward pass: {str(e)}")
            logging.error(
                f"Input shapes - x: {x.shape}, time_features: {time_features.shape}"
            )
            return None, None
