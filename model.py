import torch
import torch.nn as nn

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)


class SpamClassifier(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.embedding_dim = 128

        self.embedding = nn.Embedding(
            vocab_size,
            self.embedding_dim,
            padding_idx=0
        )

        cfg = xLSTMBlockStackConfig(

            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=4,
                )
            ),

            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=4,
                    conv1d_kernel_size=4,
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=1.3,
                    act_fn="gelu",
                ),
            ),

            context_length=50,
            embedding_dim=128,
            num_blocks=4,
            slstm_at=[1],
        )

        self.backbone = xLSTMBlockStack(cfg)

        self.classifier = nn.Linear(128, 2)

    def forward(self, x):

        x = self.embedding(x)

        x = self.backbone(x)

        x = x.mean(dim=1)

        return self.classifier(x)