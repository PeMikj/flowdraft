from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchfw.orthrus_model import (  # noqa: E402
    OrthrusConfig,
    OrthrusLM,
    diffusion_state_dict,
    initialize_diffusion_from_ar,
    set_trainable_diffusion_only,
)


class TokenBlockDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        token_ids: list[int],
        *,
        context_length: int,
        block_size: int,
        num_samples: int,
        seed: int,
    ) -> None:
        if len(token_ids) < context_length + block_size + 1:
            raise ValueError("not enough tokens for the requested context_length and block_size")
        self.token_ids = token_ids
        self.context_length = context_length
        self.block_size = block_size
        self.num_samples = num_samples
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        max_start = len(self.token_ids) - self.context_length - self.block_size
        start = self.rng.randint(0, max_start)
        ids = self.token_ids[start : start + self.context_length + self.block_size]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


class PackedSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, token_ids: list[int], *, sequence_length: int, num_samples: int, seed: int) -> None:
        if len(token_ids) < sequence_length + 1:
            raise ValueError("not enough tokens for the requested sequence_length")
        self.token_ids = token_ids
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        max_start = len(self.token_ids) - self.sequence_length
        start = self.rng.randint(0, max_start)
        ids = self.token_ids[start : start + self.sequence_length]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


def read_hf_texts(dataset_name: str, dataset_config: str | None, split: str, text_field: str, limit: int | None) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [str(row[text_field]) for row in dataset if row.get(text_field)]


def tokenize_texts(tokenizer: Any, texts: list[str], *, add_eos: bool) -> list[int]:
    token_ids: list[int] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_ids.extend(int(token_id) for token_id in ids)
        if add_eos and tokenizer.eos_token_id is not None:
            token_ids.append(int(tokenizer.eos_token_id))
    return token_ids


def normalize_to_text(row: dict[str, Any], text_field: str) -> str | None:
    if row.get(text_field):
        return str(row[text_field])
    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, list):
        chunks: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role") or message.get("from") or "user"
            content = message.get("content") or message.get("value")
            if content:
                chunks.append(f"{role}: {content}")
        if chunks:
            return "\n".join(chunks)
    for key in ("prompt", "question", "instruction"):
        if row.get(key):
            input_text = row.get("input") or row.get("context")
            answer = row.get("answer") or row.get("response") or row.get("output") or row.get("completion")
            parts = [str(row[key])]
            if input_text:
                parts.append(str(input_text))
            if answer:
                parts.append(str(answer))
            return "\n".join(parts)
    return None


def read_mixed_hf_texts(specs: list[str], text_field: str, per_dataset_limit: int | None) -> list[str]:
    texts: list[str] = []
    for spec in specs:
        parts = spec.split(":")
        dataset_name = parts[0]
        dataset_config = parts[1] if len(parts) > 1 and parts[1] else None
        split = parts[2] if len(parts) > 2 and parts[2] else "train"
        rows = read_hf_rows(dataset_name, dataset_config, split, per_dataset_limit)
        dataset_texts = [text for row in rows if (text := normalize_to_text(row, text_field))]
        texts.extend(dataset_texts)
    random.shuffle(texts)
    return texts


def read_hf_rows(dataset_name: str, dataset_config: str | None, split: str, limit: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(row) for row in dataset]


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def build_orthrus_model(
    model_name: str,
    *,
    block_size: int,
    mask_token_id: int,
    dtype: torch.dtype,
    device: str,
) -> OrthrusLM:
    from transformers import AutoConfig

    base_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config_dict = base_config.to_dict()
    config_dict["block_size"] = block_size
    config_dict["mask_token_id"] = mask_token_id
    config_dict["_attn_implementation"] = "eager"
    config = OrthrusConfig(**config_dict)
    model = OrthrusLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        ignore_mismatched_sizes=False,
    )
    initialize_diffusion_from_ar(model)
    model.to(device)
    return model


def orthrus_training_loss(
    model: OrthrusLM,
    input_ids: torch.Tensor,
    *,
    context_length: int,
    block_size: int,
    mask_token_id: int,
    temperature: float,
    hard_ce_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = input_ids.device
    context_ids = input_ids[:, :context_length]
    block_ids = input_ids[:, context_length : context_length + block_size]
    teacher_input_ids = input_ids[:, : context_length + block_size - 1]

    with torch.no_grad():
        teacher = model(input_ids=teacher_input_ids, use_cache=False, is_diffusion_pass=False)
        teacher_logits = teacher.logits[:, context_length : context_length + block_size - 1, :].detach()
        prefill = model(input_ids=context_ids, use_cache=True, is_diffusion_pass=False)
        past_key_values = prefill.past_key_values

    diff_input_ids = torch.full_like(block_ids, int(mask_token_id))
    diff_input_ids[:, 0] = block_ids[:, 0]
    position_ids = torch.arange(context_length, context_length + block_size, device=device).unsqueeze(0).expand(input_ids.shape[0], -1)
    causal_limit = torch.full((input_ids.shape[0], block_size), context_length - 1, dtype=torch.long, device=device)

    student = model(
        input_ids=diff_input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=False,
        is_diffusion_pass=True,
        causal_limit=causal_limit,
        ar_seq_len=context_length,
    )
    student_logits = student.logits[:, :-1, :]
    target_ids = block_ids[:, 1:]

    teacher_probs = (teacher_logits.float() / temperature).softmax(dim=-1)
    student_log_probs = (student_logits.float() / temperature).log_softmax(dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1).mean() * (temperature * temperature)
    ce = F.cross_entropy(student_logits.float().transpose(1, 2), target_ids)
    loss = kl + hard_ce_weight * ce
    metrics = {
        "loss": float(loss.detach().cpu()),
        "teacher_kl": float(kl.detach().cpu()),
        "hard_ce": float(ce.detach().cpu()),
    }
    return loss, metrics


def repeat_cache_for_anchors(past_key_values: Any, repeats: int) -> Any:
    if repeats == 1:
        return past_key_values
    if hasattr(past_key_values, "batch_repeat_interleave"):
        repeated = past_key_values.batch_repeat_interleave(repeats)
        return past_key_values if repeated is None else repeated
    for layer in past_key_values.layers:
        layer.keys = layer.keys.repeat_interleave(repeats, dim=0)
        layer.values = layer.values.repeat_interleave(repeats, dim=0)
    return past_key_values


def orthrus_multi_anchor_training_loss(
    model: OrthrusLM,
    input_ids: torch.Tensor,
    *,
    block_size: int,
    mask_token_id: int,
    anchors_per_sequence: int,
    min_context_length: int,
    temperature: float,
    hard_ce_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size, sequence_length = input_ids.shape
    max_anchor = sequence_length - block_size - 1
    if max_anchor < min_context_length:
        raise ValueError("sequence_length is too short for min_context_length + block_size")
    device = input_ids.device

    with torch.no_grad():
        teacher = model(input_ids=input_ids[:, :-1], use_cache=True, is_diffusion_pass=False)
        teacher_logits_all = teacher.logits.detach()
        past_key_values = teacher.past_key_values

    anchor_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    teacher_rows: list[torch.Tensor] = []
    causal_limits: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    for batch_idx in range(batch_size):
        anchors = torch.randint(
            min_context_length,
            max_anchor + 1,
            (anchors_per_sequence,),
            device=device,
        )
        for anchor in anchors.tolist():
            block = input_ids[batch_idx, anchor : anchor + block_size]
            corrupted = torch.full_like(block, int(mask_token_id))
            corrupted[0] = block[0]
            anchor_rows.append(corrupted)
            target_rows.append(block[1:])
            teacher_rows.append(teacher_logits_all[batch_idx, anchor : anchor + block_size - 1, :])
            causal_limits.append(torch.full((block_size,), anchor - 1, dtype=torch.long, device=device))
            position_rows.append(torch.arange(anchor, anchor + block_size, dtype=torch.long, device=device))

    diff_input_ids = torch.stack(anchor_rows, dim=0)
    target_ids = torch.stack(target_rows, dim=0)
    teacher_logits = torch.stack(teacher_rows, dim=0)
    causal_limit = torch.stack(causal_limits, dim=0)
    position_ids = torch.stack(position_rows, dim=0)

    # Repeat the teacher cache across sampled anchor blocks. This keeps the AR pass full-sequence
    # like Orthrus while training many diffusion blocks from one packed sequence.
    repeated_cache = repeat_cache_for_anchors(past_key_values, anchors_per_sequence)
    student = model(
        input_ids=diff_input_ids,
        position_ids=position_ids,
        past_key_values=repeated_cache,
        use_cache=False,
        is_diffusion_pass=True,
        causal_limit=causal_limit,
        ar_seq_len=sequence_length - 1,
    )
    student_logits = student.logits[:, :-1, :]

    teacher_probs = (teacher_logits.float() / temperature).softmax(dim=-1)
    student_log_probs = (student_logits.float() / temperature).log_softmax(dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1).mean() * (temperature * temperature)
    ce = F.cross_entropy(student_logits.float().transpose(1, 2), target_ids)
    loss = kl + hard_ce_weight * ce
    metrics = {
        "loss": float(loss.detach().cpu()),
        "teacher_kl": float(kl.detach().cpu()),
        "hard_ce": float(ce.detach().cpu()),
    }
    return loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-name", default="openwebtext")
    parser.add_argument("--dataset-spec", action="append", default=[])
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--dataset-limit", type=int, default=10000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-adapter", default=None)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--anchors-per-sequence", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hard-ce-weight", type=float, default=0.0)
    parser.add_argument("--model-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    mask_token_id = args.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if mask_token_id is None:
        raise ValueError("mask_token_id could not be inferred")

    dtype = resolve_dtype(args.model_dtype)
    model = build_orthrus_model(
        args.model,
        block_size=args.block_size,
        mask_token_id=int(mask_token_id),
        dtype=dtype,
        device=args.device,
    )
    if args.resume_adapter:
        resume_payload = torch.load(args.resume_adapter, map_location="cpu")
        missing, unexpected = model.load_state_dict(resume_payload["state_dict"], strict=False)
        unexpected_diff = [name for name in unexpected if "_diff" in name]
        if unexpected_diff:
            raise ValueError(f"unexpected diffusion parameters in resume checkpoint: {unexpected_diff}")
        print(
            json.dumps(
                {
                    "resume_adapter": args.resume_adapter,
                    "missing_keys": len(missing),
                    "unexpected_keys": len(unexpected),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    trainable_names = set_trainable_diffusion_only(model)
    model.train()

    if args.dataset_spec:
        texts = read_mixed_hf_texts(args.dataset_spec, args.text_field, args.dataset_limit)
    else:
        texts = read_hf_texts(args.dataset_name, args.dataset_config, args.dataset_split, args.text_field, args.dataset_limit)
    token_ids = tokenize_texts(tokenizer, texts, add_eos=True)
    if args.sequence_length is not None:
        dataset = PackedSequenceDataset(
            token_ids,
            sequence_length=args.sequence_length,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    else:
        dataset = TokenBlockDataset(
            token_ids,
            context_length=args.context_length,
            block_size=args.block_size,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_steps = max(1, int(args.steps * args.warmup_ratio))

    def lr_for_step(step_index: int) -> float:
        if step_index < warmup_steps:
            return args.learning_rate * float(step_index + 1) / float(warmup_steps)
        progress = (step_index - warmup_steps) / max(1, args.steps - warmup_steps)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()
        return args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"trainable_parameter_names": len(trainable_names), "mask_token_id": int(mask_token_id)}, sort_keys=True))
    step = 0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            input_ids = batch["input_ids"].to(args.device)
            if args.sequence_length is not None:
                loss, metrics = orthrus_multi_anchor_training_loss(
                    model,
                    input_ids,
                    block_size=args.block_size,
                    mask_token_id=int(mask_token_id),
                    anchors_per_sequence=args.anchors_per_sequence,
                    min_context_length=args.context_length,
                    temperature=args.temperature,
                    hard_ce_weight=args.hard_ce_weight,
                )
            else:
                loss, metrics = orthrus_training_loss(
                    model,
                    input_ids,
                    context_length=args.context_length,
                    block_size=args.block_size,
                    mask_token_id=int(mask_token_id),
                    temperature=args.temperature,
                    hard_ce_weight=args.hard_ce_weight,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {step + 1}: {metrics}")
            if step % args.gradient_accumulation_steps == 0:
                lr = lr_for_step(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
            (loss / args.gradient_accumulation_steps).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_((param for param in model.parameters() if param.requires_grad), 1.0)
                optimizer.step()

            step += 1
            if step % args.log_every == 0 or step == 1:
                print(json.dumps({"lr": optimizer.param_groups[0]["lr"], "step": step, **metrics}, sort_keys=True), flush=True)
            if args.save_every > 0 and step % args.save_every == 0:
                torch.save(
                    {
                        "model": args.model,
                        "block_size": args.block_size,
                        "mask_token_id": int(mask_token_id),
                        "state_dict": diffusion_state_dict(model),
                    },
                    output_path,
                )
                print(json.dumps({"checkpoint": str(output_path), "step": step}, sort_keys=True), flush=True)

    torch.save(
        {
            "model": args.model,
            "block_size": args.block_size,
            "mask_token_id": int(mask_token_id),
            "state_dict": diffusion_state_dict(model),
        },
        output_path,
    )
    print(json.dumps({"checkpoint": str(output_path), "steps": step}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
