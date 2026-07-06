from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchfw.cfm_drafter import build_cfm_drafter_from_model
from benchfw.model_loading import load_model_and_tokenizer
from benchfw.config import ModelConfig


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


def read_prompt_texts(path: str | Path) -> list[str]:
    texts: list[str] = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("prompt") or row.get("text")
            if text:
                texts.append(str(text))
    return texts


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


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--drafter-hidden-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--teacher-kl-weight", type=float, default=1.0)
    parser.add_argument("--hard-ce-weight", type=float, default=0.25)
    parser.add_argument("--flow-consistency-weight", type=float, default=0.1)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model_and_tokenizer(
        ModelConfig(
            name_or_path=args.model,
            tokenizer_name_or_path=args.tokenizer,
            dtype=args.model_dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        )
    )
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    if args.prompt_file:
        texts = read_prompt_texts(args.prompt_file)
    elif args.dataset_name:
        texts = read_hf_texts(
            args.dataset_name,
            args.dataset_config,
            args.dataset_split,
            args.text_field,
            args.dataset_limit,
        )
    else:
        raise ValueError("provide either --prompt-file or --dataset-name")

    token_ids = tokenize_texts(tokenizer, texts, add_eos=True)
    dataset = TokenBlockDataset(
        token_ids,
        context_length=args.context_length,
        block_size=args.block_size,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    drafter = build_cfm_drafter_from_model(
        model,
        block_size=args.block_size,
        drafter_hidden_size=args.drafter_hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        teacher_kl_weight=args.teacher_kl_weight,
        hard_ce_weight=args.hard_ce_weight,
        flow_consistency_weight=args.flow_consistency_weight,
        distill_temperature=args.distill_temperature,
    )
    device = next(model.parameters()).device
    train_dtype = resolve_dtype(args.model_dtype) if args.model_dtype != "float32" else torch.float32
    drafter.to(device=device, dtype=train_dtype)
    drafter.train()

    optimizer = torch.optim.AdamW(drafter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            input_ids = batch["input_ids"].to(device)
            target_ids = input_ids[:, args.context_length : args.context_length + args.block_size]

            with torch.inference_mode():
                teacher_ids = input_ids[:, : args.context_length + args.block_size - 1]
                teacher = model(input_ids=teacher_ids, output_hidden_states=True, use_cache=False)
                context_hidden = teacher.hidden_states[-1][:, args.context_length - 1, :].detach()
                teacher_logits = teacher.logits[
                    :,
                    args.context_length - 1 : args.context_length - 1 + args.block_size,
                    :,
                ].detach()

            loss, metrics = drafter.training_loss(
                context_hidden=context_hidden,
                target_ids=target_ids,
                teacher_logits=teacher_logits,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step % args.log_every == 0 or step == 1:
                print(json.dumps({"step": step, **metrics}, sort_keys=True))

    torch.save(drafter.checkpoint_payload(), output_path)
    print(json.dumps({"checkpoint": str(output_path), "steps": step}, sort_keys=True))


if __name__ == "__main__":
    main()
