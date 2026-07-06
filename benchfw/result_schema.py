from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    mode_name: str
    prompt_id: str
    run_index: int
    prompt_metadata: dict[str, Any]
    generated_token_ids: list[int]
    generated_only_token_ids: list[int]
    decoded_text: str
    generated_only_text: str
    generation_time_s: float
    input_length: int
    output_length: int
    peak_gpu_memory_bytes: int | None = None
    internal_metrics: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        generated_tokens = max(0, self.output_length - self.input_length)
        tokens_per_second = generated_tokens / self.generation_time_s if self.generation_time_s > 0 else None
        ms_per_token = (self.generation_time_s * 1000.0 / generated_tokens) if generated_tokens else None
        row = asdict(self)
        row.update(
            {
                "category": self.prompt_metadata.get("category"),
                "dataset": self.prompt_metadata.get("dataset"),
                "length_bucket": self.prompt_metadata.get("length_bucket"),
                "generated_tokens": generated_tokens,
                "tokens_per_second": tokens_per_second,
                "milliseconds_per_token": ms_per_token,
                "acceptance_length": self.internal_metrics.get("acceptance_length"),
                "acceptance_length_distribution": self.internal_metrics.get("acceptance_length_distribution"),
                "tpf": self.internal_metrics.get("tpf"),
                "verifier_forward_passes": self.internal_metrics.get("verifier_forward_passes"),
                "drafter_forward_passes": self.internal_metrics.get("drafter_forward_passes"),
                "total_forward_passes": self.internal_metrics.get("total_forward_passes"),
                "decode_forward_passes": self.internal_metrics.get("decode_forward_passes"),
                "tpf_including_prefill": self.internal_metrics.get("tpf_including_prefill"),
                "generated_tokens_for_tpf": self.internal_metrics.get("generated_tokens_for_tpf"),
                "draft_block_size": self.internal_metrics.get("draft_block_size"),
                "block_size": self.internal_metrics.get("draft_block_size"),
            }
        )
        return row
