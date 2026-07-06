from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CfmDrafterConfig:
    vocab_size: int
    hidden_size: int
    block_size: int = 32
    drafter_hidden_size: int = 1024
    num_layers: int = 2
    dropout: float = 0.0
    label_smoothing: float = 0.0
    teacher_kl_weight: float = 1.0
    hard_ce_weight: float = 0.25
    flow_consistency_weight: float = 0.1
    distill_temperature: float = 1.0


def view_for(tensor: Tensor, target: Tensor) -> Tensor:
    return tensor.view(*tensor.shape, *([1] * (target.ndim - tensor.ndim)))


class CategoricalFlowMapDrafter(nn.Module):
    """Conditional CFM drafter for block proposals after a frozen AR context.

    The CFM state lives in the relaxed one-hot simplex `[B, K, V]`. A frozen
    token embedding matrix maps that state into the AR hidden space, then a
    small trainable block network predicts endpoint logits for all K positions.
    """

    def __init__(
        self,
        config: CfmDrafterConfig,
        *,
        token_embedding_weight: Tensor | None = None,
        output_embedding_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.context_proj = nn.Linear(config.hidden_size, config.drafter_hidden_size)
        self.state_proj = nn.Linear(config.hidden_size, config.drafter_hidden_size)
        self.position_embedding = nn.Embedding(config.block_size, config.drafter_hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(2, config.drafter_hidden_size),
            nn.SiLU(),
            nn.Linear(config.drafter_hidden_size, config.drafter_hidden_size),
        )
        layers: list[nn.Module] = []
        for _ in range(config.num_layers):
            layers.extend(
                [
                    nn.LayerNorm(config.drafter_hidden_size),
                    nn.Linear(config.drafter_hidden_size, config.drafter_hidden_size * 4),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.drafter_hidden_size * 4, config.drafter_hidden_size),
                ]
            )
        self.net = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(config.drafter_hidden_size)
        self.output_proj = nn.Linear(config.drafter_hidden_size, config.vocab_size, bias=False)

        if token_embedding_weight is None:
            self.input_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.input_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.input_embedding.weight.detach().copy_(token_embedding_weight.detach().to(self.input_embedding.weight))
            self.input_embedding.weight.requires_grad_(False)

        if output_embedding_weight is not None:
            if output_embedding_weight.shape == self.output_proj.weight.shape:
                self.output_proj.weight.detach().copy_(output_embedding_weight.detach().to(self.output_proj.weight))

    def _state_embedding(self, xt: Tensor) -> Tensor:
        return xt.to(self.input_embedding.weight.dtype) @ self.input_embedding.weight

    def prior(self, batch_size: int, device: torch.device | str) -> Tensor:
        cats = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=(batch_size, self.config.block_size),
            device=device,
        )
        return F.one_hot(cats, num_classes=self.config.vocab_size).to(dtype=self.input_embedding.weight.dtype)

    def interpolate(self, x0: Tensor, target_ids: Tensor, t: Tensor) -> Tensor:
        xt = x0 * (1.0 - view_for(t, x0))
        xt.scatter_add_(
            -1,
            target_ids[..., None],
            view_for(t, target_ids[..., None]).to(xt.dtype).expand_as(target_ids[..., None]),
        )
        return xt

    def forward_logits(self, context_hidden: Tensor, xt: Tensor, s: Tensor, t: Tensor) -> Tensor:
        batch_size, block_size, _ = xt.shape
        positions = torch.arange(block_size, device=xt.device)
        hidden = (
            self.context_proj(context_hidden).unsqueeze(1)
            + self.state_proj(self._state_embedding(xt))
            + self.position_embedding(positions).unsqueeze(0)
            + self.time_mlp(torch.stack([s, t], dim=-1)).unsqueeze(1)
        )
        for idx in range(0, len(self.net), 5):
            residual = hidden
            x = self.net[idx](hidden)
            x = self.net[idx + 1](x)
            x = self.net[idx + 2](x)
            x = self.net[idx + 3](x)
            x = self.net[idx + 4](x)
            hidden = residual + x
        return self.output_proj(self.final_norm(hidden))

    def xst(self, context_hidden: Tensor, x: Tensor, s: Tensor, t: Tensor) -> Tensor:
        logits = self.forward_logits(context_hidden, x, s, t)
        probs = logits.softmax(dim=-1)
        dt = (t - s) / (1.0 - s + 1e-8)
        return x + view_for(dt, x) * (probs - x)

    def training_loss(
        self,
        *,
        context_hidden: Tensor,
        target_ids: Tensor,
        teacher_logits: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        batch_size = target_ids.shape[0]
        x0 = self.prior(batch_size, target_ids.device)
        t = torch.rand(batch_size, device=target_ids.device)
        xt = self.interpolate(x0, target_ids, t)
        logits = self.forward_logits(context_hidden, xt, t, t)

        temperature = self.config.distill_temperature
        teacher_probs = (teacher_logits / temperature).softmax(dim=-1)
        log_probs = (logits / temperature).log_softmax(dim=-1)
        kl = F.kl_div(log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)
        ce = F.cross_entropy(
            logits.transpose(1, 2),
            target_ids,
            label_smoothing=self.config.label_smoothing,
        )

        s = torch.rand(batch_size, device=target_ids.device) * t
        xs = self.interpolate(x0, target_ids, s)
        xst = self.xst(context_hidden, xs, s, t)
        with torch.no_grad():
            endpoint = self.forward_logits(context_hidden, xst, t, t).softmax(dim=-1)
        consistency_logits = self.forward_logits(context_hidden, xs, s, t)
        consistency = F.kl_div(
            consistency_logits.log_softmax(dim=-1),
            endpoint,
            reduction="batchmean",
        )

        loss = (
            self.config.teacher_kl_weight * kl
            + self.config.hard_ce_weight * ce
            + self.config.flow_consistency_weight * consistency
        )
        metrics = {
            "loss": float(loss.detach().cpu()),
            "teacher_kl": float(kl.detach().cpu()),
            "hard_ce": float(ce.detach().cpu()),
            "flow_consistency": float(consistency.detach().cpu()),
        }
        return loss, metrics

    @torch.inference_mode()
    def sample(self, context_hidden: Tensor, *, sampling_steps: int = 1) -> Tensor:
        x = self.prior(context_hidden.shape[0], context_hidden.device)
        ts = torch.linspace(0.0, 1.0, sampling_steps + 1, device=context_hidden.device)
        for s_scalar, t_scalar in zip(ts[:-1], ts[1:]):
            s = s_scalar.expand(context_hidden.shape[0])
            t = t_scalar.expand(context_hidden.shape[0])
            x = self.xst(context_hidden, x, s, t)
        return x.argmax(dim=-1)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "state_dict": self.state_dict()}


def build_cfm_drafter_from_model(
    model: Any,
    *,
    block_size: int,
    drafter_hidden_size: int = 1024,
    num_layers: int = 2,
    dropout: float = 0.0,
    label_smoothing: float = 0.0,
    teacher_kl_weight: float = 1.0,
    hard_ce_weight: float = 0.25,
    flow_consistency_weight: float = 0.1,
    distill_temperature: float = 1.0,
) -> CategoricalFlowMapDrafter:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    hidden_size = int(getattr(model.config, "hidden_size", input_embeddings.embedding_dim))
    vocab_size = int(getattr(model.config, "vocab_size", input_embeddings.num_embeddings))
    config = CfmDrafterConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        block_size=block_size,
        drafter_hidden_size=drafter_hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        label_smoothing=label_smoothing,
        teacher_kl_weight=teacher_kl_weight,
        hard_ce_weight=hard_ce_weight,
        flow_consistency_weight=flow_consistency_weight,
        distill_temperature=distill_temperature,
    )
    return CategoricalFlowMapDrafter(
        config,
        token_embedding_weight=input_embeddings.weight,
        output_embedding_weight=getattr(output_embeddings, "weight", None),
    )


def load_cfm_drafter(path: str, *, device: torch.device | str, dtype: torch.dtype | None = None) -> CategoricalFlowMapDrafter:
    payload = torch.load(path, map_location="cpu")
    config = CfmDrafterConfig(**payload["config"])
    drafter = CategoricalFlowMapDrafter(config)
    drafter.load_state_dict(payload["state_dict"])
    drafter.to(device=device)
    if dtype is not None:
        drafter.to(dtype=dtype)
    drafter.eval()
    return drafter
