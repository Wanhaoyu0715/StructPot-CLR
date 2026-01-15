import logging
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class CLIPConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)



class PointNetConfig:
    """ base PointNet config """

    def __init__(self, embeddingSize, numberofPoints,
                 **kwargs):
        self.embeddingSize = embeddingSize
        self.numberofPoints = numberofPoints  # number of points


        for k, v in kwargs.items():
            setattr(self, k, v)








class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        """
        Positional encoding for transformer.

        Args:
            d_model: Embedding dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x


class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
class seq_emb(nn.Module):
    def __init__(self, input_channels=1, embedding_dim=128, num_blocks=[2, 2, 2, 2]):
        super(seq_emb, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, num_blocks[0])
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock1D.expansion, embedding_dim)
    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * BasicBlock1D.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * BasicBlock1D.expansion),
            )
        layers = []
        layers.append(BasicBlock1D(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock1D.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CLIP(nn.Module):
    def __init__(self, config, pointNetConfig=None, cry_encoder=None):
        super().__init__()

        self.config = config
        self.pointNetConfig = pointNetConfig
        self.pointNet = None
        self.cry_encoder = cry_encoder
        embeddingSize = config.n_embd
        self.block_size = config.block_size

        self.pos_emb_wf = PositionalEncoding(embeddingSize, dropout=0.1, max_len=self.pointNetConfig.numberofPoints)
        self.penalty_labels = torch.eye(config.block_size)
        self.wf_emb = seq_emb(embedding_dim=config.n_embd)

        self.mlp_predict = nn.Sequential(
            nn.Linear(2 * config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(config.n_embd, 64),
            nn.GELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(64, pointNetConfig.numberofPoints),
        )

        self.fc_project_formula = nn.Linear(config.n_embd, config.n_embd)
        self.fc_project_wf = nn.Linear(config.n_embd, config.n_embd)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer('total_labels', torch.arange(30000))
        
        self.rng = torch.Generator()
        self.rng.manual_seed(42)

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.ln_wf = nn.LayerNorm(config.n_embd)

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.GRU)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.startswith('gru1.'):
                    decay.add(fpn)
                elif pn.startswith('gru2.'):
                    decay.add(fpn)
                elif pn.endswith('grid'):
                    no_decay.add(fpn)
                elif mn.endswith('cope'):
                    no_decay.add(fpn)

        no_decay.add('logit_scale')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, \
            f"parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"


        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, batch):
        wf = batch['wf']
        b, _ = batch['nodes'].size()

        x = self.cry_encoder(batch)
        formula_embedding_final = self.fc_project_formula(x)
        formula_embedding_final = formula_embedding_final / formula_embedding_final.norm(dim=-1, keepdim=True)

        wf_embeddings_final = self.wf_emb(wf)
        wf_embeddings_final = self.ln_wf(wf_embeddings_final)
        wf_embeddings_final = self.fc_project_wf(wf_embeddings_final)
        wf_embeddings_final = wf_embeddings_final / wf_embeddings_final.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_wf = logit_scale * wf_embeddings_final @ formula_embedding_final.t()
        logits_per_formula = logits_per_wf.t()

        labels = self.total_labels[:b]

        if wf.shape[0] == b:
            loss = (F.cross_entropy(logits_per_wf, labels) +
                    F.cross_entropy(logits_per_formula, labels)) / 2
        else:
            loss = 0.0

        return loss, logits_per_wf


