from __future__ import annotations

import torch
from torch import nn


SCALES = 3
SOURCES = 22
STATS = 10
TASKS = 7
HORIZONS = 3
D_MODEL = 64
PRIVATE_EXPERTS = 4
TOP_K = 2
VARIANTS = (
    "no_source_mixer",
    "single_expert",
)


class Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(D_MODEL, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, D_MODEL),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class SSTHMoEAblation(nn.Module):
    """One-factor CA84 ablation with the original 3 x 22 x 10 I/O contract."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown CA84P ablation: {variant}")
        self.variant = variant
        self.register_buffer("feature_mean", torch.zeros(SCALES, SOURCES, STATS))
        self.register_buffer("feature_scale", torch.ones(SCALES, SOURCES, STATS))
        self.tokenizer = nn.Sequential(
            nn.Linear(STATS, D_MODEL),
            nn.LayerNorm(D_MODEL),
        )
        self.source_embedding = nn.Parameter(torch.empty(SOURCES, D_MODEL))
        self.scale_embedding = nn.Parameter(torch.empty(SCALES, D_MODEL))
        nn.init.normal_(self.source_embedding, std=0.02)
        nn.init.normal_(self.scale_embedding, std=0.02)

        if variant == "no_source_mixer":
            self.source_mixer: nn.Module = nn.Identity()
        else:
            source_layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=4,
                dim_feedforward=192,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.source_mixer = nn.TransformerEncoder(
                source_layer, num_layers=2, enable_nested_tensor=False
            )

        scale_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.scale_mixer = nn.TransformerEncoder(
            scale_layer, num_layers=1, enable_nested_tensor=False
        )

        self.task_query = nn.Parameter(torch.empty(TASKS, D_MODEL))
        self.horizon_query = nn.Parameter(torch.empty(HORIZONS, D_MODEL))
        self.interaction_query = nn.Parameter(
            torch.zeros(TASKS, HORIZONS, D_MODEL)
        )
        nn.init.normal_(self.task_query, std=0.02)
        nn.init.normal_(self.horizon_query, std=0.02)

        self.query_attention = nn.MultiheadAttention(
            D_MODEL, 4, dropout=0.1, batch_first=True
        )
        self.query_norm = nn.LayerNorm(D_MODEL)
        self.shared_expert = Expert()
        if variant == "single_expert":
            self.private_experts = nn.ModuleList()
            self.router = None
        else:
            self.private_experts = nn.ModuleList(
                Expert() for _ in range(PRIVATE_EXPERTS)
            )
            self.router: nn.Module | None = nn.Linear(D_MODEL, PRIVATE_EXPERTS)
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.task_heads = nn.ModuleList(
            nn.Linear(D_MODEL, 1) for _ in range(TASKS)
        )

    def set_scaler(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        if tuple(mean.shape) != (SCALES, SOURCES, STATS):
            raise ValueError("CA84P feature mean geometry changed")
        if tuple(scale.shape) != tuple(mean.shape):
            raise ValueError("CA84P feature scale geometry changed")
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(scale.clamp_min(1.0e-6))

    def _physiology_tokens(self, values: torch.Tensor) -> torch.Tensor:
        if tuple(values.shape[1:]) != (SCALES, SOURCES, STATS):
            raise ValueError("CA84P input geometry changed")
        values = (values - self.feature_mean) / self.feature_scale
        tokens = self.tokenizer(values)
        tokens = (
            tokens
            + self.source_embedding[None, None, :, :]
            + self.scale_embedding[None, :, None, :]
        )
        batch = tokens.shape[0]
        tokens = self.source_mixer(
            tokens.reshape(batch * SCALES, SOURCES, D_MODEL)
        )
        tokens = tokens.reshape(batch, SCALES, SOURCES, D_MODEL)
        tokens = tokens.permute(0, 2, 1, 3).contiguous()
        tokens = self.scale_mixer(
            tokens.reshape(batch * SOURCES, SCALES, D_MODEL)
        )
        return tokens.reshape(batch, SOURCES * SCALES, D_MODEL)

    def _queries(self) -> torch.Tensor:
        return (
            self.task_query[:, None, :]
            + self.horizon_query[None, :, :]
            + self.interaction_query
        ).reshape(TASKS * HORIZONS, D_MODEL)

    def _route(
        self, representations: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.variant == "single_expert":
            probability = torch.zeros(
                (*representations.shape[:-1], PRIVATE_EXPERTS),
                dtype=representations.dtype,
                device=representations.device,
            )
            probability[..., 0] = 1.0
            return torch.zeros_like(representations), {
                "router_probability": probability,
                "router_top1": torch.zeros(
                    representations.shape[:-1],
                    dtype=torch.long,
                    device=representations.device,
                ),
                "router_balance_loss": representations.new_zeros(()),
                "router_entropy": representations.new_zeros(()),
            }
        private = torch.stack(
            [expert(representations) for expert in self.private_experts], dim=2
        )
        probability = torch.softmax(self.router(representations), dim=-1)
        top_probability, top_index = torch.topk(
            probability, k=TOP_K, dim=-1
        )
        top_probability = top_probability / top_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        sparse_weight = torch.zeros_like(probability).scatter(
            -1, top_index, top_probability
        )
        routed = (private * sparse_weight.unsqueeze(-1)).sum(dim=2)
        top1 = top_index[..., 0]
        mean_probability = probability.mean(dim=(0, 1))
        mean_top1 = torch.nn.functional.one_hot(
            top1, num_classes=PRIVATE_EXPERTS
        ).to(probability.dtype).mean(dim=(0, 1))
        balance_loss = (
            mean_probability - 1.0 / PRIVATE_EXPERTS
        ).square().mean()
        balance_loss = balance_loss + (
            mean_top1 - 1.0 / PRIVATE_EXPERTS
        ).square().mean()
        entropy = -(
            probability * probability.clamp_min(1.0e-8).log()
        ).sum(dim=-1).mean()
        return routed, {
            "router_probability": probability,
            "router_top1": top1,
            "router_balance_loss": balance_loss,
            "router_entropy": entropy,
        }

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        physiology = self._physiology_tokens(values)
        queries = self._queries().unsqueeze(0).expand(values.shape[0], -1, -1)
        attended, attention = self.query_attention(
            queries, physiology, physiology, need_weights=True
        )
        representations = self.query_norm(queries + attended)
        private, auxiliary = self._route(representations)
        routed = self.output_norm(
            representations + self.shared_expert(representations) + private
        )
        routed = routed.reshape(values.shape[0], TASKS, HORIZONS, D_MODEL)
        logits = torch.empty(
            values.shape[0],
            TASKS,
            HORIZONS,
            dtype=values.dtype,
            device=values.device,
        )
        for task, head in enumerate(self.task_heads):
            logits[:, task, :] = head(routed[:, task, :, :]).squeeze(-1)
        auxiliary["query_attention"] = attention
        return logits, auxiliary
