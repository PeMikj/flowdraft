from __future__ import annotations

from typing import Any

from .config import ModelConfig


def patch_transformers_masking_utils() -> None:
    try:
        import inspect
        import transformers.modeling_utils as modeling_utils
        import transformers.masking_utils as masking_utils
    except Exception:
        return

    attention_functions = getattr(modeling_utils, "ALL_ATTENTION_FUNCTIONS", None)
    if attention_functions is not None and not hasattr(attention_functions, "get_interface"):
        def get_interface(name: str, default: Any | None = None) -> Any:
            if name == "eager" and default is not None:
                return default
            try:
                return attention_functions[name]
            except KeyError:
                if default is not None:
                    return default
                raise

        attention_functions.get_interface = get_interface

    original = getattr(masking_utils, "create_causal_mask", None)
    if original is None:
        return
    signature = inspect.signature(original)
    if "inputs_embeds" in signature.parameters or getattr(original, "_flowdraft_alias_patch", False):
        return

    def create_causal_mask_alias(*args: Any, **kwargs: Any) -> Any:
        input_embeds = kwargs.get("input_embeds")
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            input_embeds = kwargs.pop("inputs_embeds")
            kwargs["input_embeds"] = input_embeds
        if "cache_position" not in kwargs and input_embeds is not None:
            import torch

            position_ids = kwargs.get("position_ids")
            if position_ids is not None:
                kwargs["cache_position"] = position_ids[0].to(device=input_embeds.device, dtype=torch.long)
            else:
                kwargs["cache_position"] = torch.arange(
                    input_embeds.shape[1],
                    device=input_embeds.device,
                    dtype=torch.long,
                )
        return original(*args, **kwargs)

    create_causal_mask_alias._flowdraft_alias_patch = True
    masking_utils.create_causal_mask = create_causal_mask_alias


def resolve_torch_dtype(dtype_name: str) -> Any:
    import torch

    mapping = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def load_model_and_tokenizer(config: ModelConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    patch_transformers_masking_utils()

    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(config.dtype),
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation

    tokenizer_name = config.tokenizer_name_or_path or config.name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=config.trust_remote_code)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=config.trust_remote_code,
            extra_special_tokens={},
        )
    model = AutoModelForCausalLM.from_pretrained(config.name_or_path, **model_kwargs)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer
