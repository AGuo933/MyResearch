import torch
import torch.nn as nn
from models.encoder import Encoder, EncoderLayer
from models.attn import ProbAttention, AttentionLayer
from models.embed import DataEmbedding


class Informer(nn.Module):
    def __init__(
        self,
        enc_in=17,  # 17个输入特征(16个原始特征 + 1个时间特征)
        d_model=128,  # 嵌入维度
        n_heads=8,  # 注意力头数
        e_layers=3,  # encoder层数
        d_ff=512,  # 前馈网络维度
        dropout=0.2,
        activation="gelu",
    ):
        super(Informer, self).__init__()

        # Encoding
        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed_type="timeF", freq="h", dropout=dropout
        )

        # Attention
        Attn = ProbAttention

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(
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
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        # Output projection
        self.fc = nn.Linear(d_model, 1)  # 最终输出为1维

    def forward(self, x, time_features, enc_self_mask=None):
        # x shape: [batch_size, features]
        # time_features shape: [batch_size, 3] (month, weekday, day)

        # 添加序列维度
        x = x.unsqueeze(1)  # [batch_size, 1, features]
        time_mark = time_features.unsqueeze(1)  # [batch_size, 1, 3]

        # Embedding
        enc_out = self.enc_embedding(x, time_mark)

        # 后续处理保持不变
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = enc_out.reshape(enc_out.size(0), -1)
        output = self.fc(enc_out)

        return output
