"""
Online inference pipeline.

Runs a forward pass on a user-provided (image, question) pair with
probing hooks active to capture:
  - hidden-state activations per generated token
  - cross-attention weights to image patch tokens per step

Works with Qwen2.5-VL, Monet-7B, and LVR (all share the same
Qwen2.5-VL architecture; differences are in weights and trigger tokens).
"""

from __future__ import annotations

import re
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from PIL import Image
from qwen_vl_utils import process_vision_info

_model_cache: dict = {}
_processor_cache: dict = {}

MODEL_IDS = {
    "qwen": "model/Qwen2.5-VL-7B-Instruct",
    "monet": "model/Monet-7B",
    "lvr": "model/LVR-7B",
}

MAX_PIXELS = 512 * 512

LATENT_MARKERS = (
    "<abs_vis_token>",       # Monet: opens a latent visual span.
    "<abs_vis_token_pad>",   # Monet: occupies a latent visual position.
    "</abs_vis_token>",      # Monet: closes a latent visual span.
    "<|lvr_start|>",         # LVR: opens a latent reasoning block.
    "<|lvr|>",               # LVR: occupies a latent reasoning position.
    "<|lvr_latent_end|>",    # LVR: ends the latent phase.
    "<|lvr_end|>",           # LVR: closes the reasoning block.
)

def _get_model_and_processor(model_name: str):
    """Load (and cache) a model + processor by name."""
    if model_name not in _model_cache:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from backend.latent_model import (
            LEGACY_QWEN_KEY_MAPPING,
            LatentAwareQwen2_5_VLForConditionalGeneration,
            validate_checkpoint_load,
        )

        model_id = MODEL_IDS[model_name]
        print(f"[inference] Loading {model_id} ...")

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model_class = (
            Qwen2_5_VLForConditionalGeneration
            if model_name == "qwen"
            else LatentAwareQwen2_5_VLForConditionalGeneration
        )
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if model_name == "qwen":
            model = model_class.from_pretrained(model_id, **load_kwargs)
        else:
            model, loading_info = model_class.from_pretrained(
                model_id,
                key_mapping=LEGACY_QWEN_KEY_MAPPING,
                output_loading_info=True,
                **load_kwargs,
            )
            validate_checkpoint_load(loading_info, model_id)
        if model_name != "qwen":
            model.configure_latent_decoding(model_name)
            spec = model.latent_decoding
            print(
                f"[inference] Enabled {model_name} latent decoding "
                f"(start={spec.start_id}, latent={spec.placeholder_id}, end={spec.end_id})"
            )
        model.eval()
        _model_cache[model_name] = model
        _processor_cache[model_name] = processor

    return _model_cache[model_name], _processor_cache[model_name]


# ---------------------------------------------------------------------------
# Hook management
# ---------------------------------------------------------------------------

class ActivationStore:
    """Collects attention weights injected by forward hooks, plus hidden states
    extracted from generate(output_hidden_states=True)."""

    def __init__(self, prompt_length: int = 0):
        self.prompt_length = prompt_length
        self.hidden_states: list[np.ndarray] = []
        self.attn_weights: list[np.ndarray] = []
        self.layer_attn_weights: list[list[np.ndarray | None]] = []
        self._hooks: list = []

    def clear(self):
        self.hidden_states.clear()
        self.attn_weights.clear()
        self.layer_attn_weights.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _register_hooks(model, store: ActivationStore, target_layer: int = -1):
    """
    Register attention hooks on every decoder layer.

    All layers retain attention to generated positions for the token matrix.
    The target layer additionally retains its full attention vector for the
    existing image-patch attention visualization.

    Hidden states are NOT captured via a hook anymore: a hook on layers[-1]
    sees the pre-final-norm output, whereas the corpus stores the post-norm
    hidden_states[-1]. To stay on-manifold, hidden states are pulled from
    generate(output_hidden_states=True) in run_inference instead.
    """
    layers = model.model.language_model.layers
    target_index = target_layer % len(layers)
    store.layer_attn_weights = [[] for _ in layers]

    def _make_attn_hook(layer_index: int):
        def _attn_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                weights = output[1][0, :, -1, :].mean(0).detach().float().cpu().numpy()
                generated_weights = weights[store.prompt_length:].astype(np.float16)
                store.layer_attn_weights[layer_index].append(generated_weights)
                if layer_index == target_index:
                    store.attn_weights.append(weights)
            else:
                store.layer_attn_weights[layer_index].append(None)
                if layer_index == target_index:
                    store.attn_weights.append(None)
        return _attn_hook

    for layer_index, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            store._hooks.append(
                layer.self_attn.register_forward_hook(_make_attn_hook(layer_index))
            )


# ---------------------------------------------------------------------------
# Token type classification
# ---------------------------------------------------------------------------
def _classify_token_types(
    token_ids: list[int],
    processor,
    num_image_tokens: int,
    model_name: str,
) -> list[str]:
    """
    Per-token labels: 'text' | 'visual' | 'latent'.
    Boundary and continuous-placeholder positions in a latent span are latent.
    """
    tokenizer = processor.tokenizer
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id   = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    types: list[str] = []
    in_visual = False
    in_latent = False
    if model_name == "monet":
        latent_start = tokenizer.convert_tokens_to_ids("<abs_vis_token>")
        latent_end = tokenizer.convert_tokens_to_ids("</abs_vis_token>")
    elif model_name == "lvr":
        latent_start = tokenizer.convert_tokens_to_ids("<|lvr_start|>")
        latent_end = tokenizer.convert_tokens_to_ids("<|lvr_end|>")
    else:
        latent_start = latent_end = None

    for tid in token_ids:
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        if tid == vision_start_id:
            in_visual = True
            types.append("visual")
        elif tid == vision_end_id:
            in_visual = False
            types.append("visual")
        elif in_visual:
            types.append("visual")
        elif tid == latent_start:
            in_latent = True
            types.append("latent")
        elif in_latent:
            types.append("latent")
            if tid == latent_end:
                in_latent = False
        elif any(marker in tok_str for marker in LATENT_MARKERS):
            types.append("latent")
        else:
            types.append("text")
    return types

# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_inference(
    image: Image.Image,
    question: str,
    model_name: str,
    max_new_tokens: int = 512,
    attention_intervention: Optional[str] = None,
    prefix_ids: Optional[list[int]] = None,
    prefix_activations: Optional[list[np.ndarray]] = None,
    prefix_attn: Optional[list[np.ndarray]] = None,
    prefix_layer_attn: Optional[list[list[np.ndarray]]] = None,
) -> dict:
    """
    Run a forward pass and return all extracted data needed by the dashboard.

    Parameters
    ----------
    image           : PIL Image
    question        : user question string
    model_name      : 'qwen' | 'monet' | 'lvr'
    max_new_tokens  : generation budget
    attention_intervention : optional causal attention intervention mode
    prefix_ids      : if set, prepend these token ids (for edited-instance continuation)
    prefix_activations : activations from the prefix (editing case)
    prefix_attn        : attention weights from the prefix (editing case)

    Returns
    -------
    dict with keys:
        activations    : np.ndarray (T, D)
        attn_weights   : list[np.ndarray | None]
        token_types    : list[str]  length T
        token_strings  : list[str]  length T
        token_ids      : list[int]  length T
        generated_text : str
        image_grid_hw  : tuple[int, int]
        image_token_range : tuple[int, int]
    """
    model, processor = _get_model_and_processor(model_name)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "max_pixels": MAX_PIXELS},
                {"type": "text",  "text": question},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    img_w, img_h = image.size
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        merge_size = getattr(processor.image_processor, "merge_size", 2)
        _, grid_h, grid_w = image_grid_thw[0].tolist()
        grid_h //= merge_size
        grid_w //= merge_size
    else:
        grid_h = img_h // 28   # 14px patch × 2 merge
        grid_w = img_w // 28

    input_ids_list = inputs["input_ids"][0].tolist()
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id   = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    try:
        img_token_start = input_ids_list.index(vision_start_id) + 1
        img_token_end   = input_ids_list.index(vision_end_id)
    except ValueError:
        img_token_start = 0
        img_token_end   = grid_h * grid_w
    print(f"[debug] tokens in range: {img_token_end - img_token_start}, grid_h*grid_w: {grid_h * grid_w}")

    input_len = inputs["input_ids"].shape[1]
    store = ActivationStore(prompt_length=input_len)
    _register_hooks(model, store)

    if attention_intervention:
        if model_name == "qwen" or not hasattr(model, "latent_attention_intervention"):
            store.remove_hooks()
            raise ValueError(
                "Latent attention interventions require Monet or LVR; "
                "the baseline Qwen model has no generated visual latent tokens."
            )
        if attention_intervention != "question_latent_answer_bottleneck":
            store.remove_hooks()
            raise ValueError(f"Unknown attention intervention: {attention_intervention}")
        model.latent_attention_intervention = {
            "mode": attention_intervention,
            "prompt_length": input_len,
            "image_token_range": (img_token_start, img_token_end),
        }

    print(f"[inference] Running generation for {model_name} ...")
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_attentions=True,        # populates attn weights for the hook
                output_hidden_states=True,     # post-norm hidden states, per step
                return_dict_in_generate=True,
            )
    finally:
        if attention_intervention and hasattr(model, "latent_attention_intervention"):
            model.latent_attention_intervention = None
        store.remove_hooks()
    print(f"[inference] Finished generation for {model_name} ...")

    sequences = outputs.sequences
    gen_ids = sequences[0, input_len:].tolist()

    for step_hs in outputs.hidden_states:
        last_layer = step_hs[-1]
        if last_layer.ndim != 3:
            continue
        vec = last_layer[0, -1, :].detach().float().cpu().numpy().astype(np.float16)
        store.hidden_states.append(vec)
    store.hidden_states = store.hidden_states[:len(gen_ids)]

    token_strings = [
        processor.tokenizer.decode([tid], skip_special_tokens=False)
        for tid in gen_ids
    ]
    generated_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

    all_activations   = (prefix_activations or []) + store.hidden_states
    all_attn          = (prefix_attn or [])        + store.attn_weights
    if prefix_layer_attn:
        layer_attn_weights = [
            list(prefix_layer_attn[i]) + layer_steps
            if i < len(prefix_layer_attn) else layer_steps
            for i, layer_steps in enumerate(store.layer_attn_weights)
        ]
    else:
        layer_attn_weights = store.layer_attn_weights
    all_ids           = (prefix_ids or [])         + gen_ids

    token_types = _classify_token_types(all_ids, processor, grid_h * grid_w, model_name)

    if all_activations:
        activations_array = np.stack(all_activations).astype(np.float32)
    else:
        activations_array = np.empty((0, model.config.hidden_size), dtype=np.float32)

    return {
        "activations":    activations_array,
        "attn_weights":   all_attn,
        "layer_attn_weights": layer_attn_weights,
        "attention_layer_count": len(layer_attn_weights),
        "token_types":    token_types,
        "token_strings":  token_strings,
        "token_ids":      all_ids,
        "generated_text": generated_text,
        "prompt_length": input_len,
        "image_grid_hw":  (grid_h, grid_w),
        "image_token_range": (img_token_start, img_token_end),
        "attention_intervention": attention_intervention,
    }


def run_edited_inference(
    image: Image.Image,
    question: str,
    model_name: str,
    original_result: dict,
    edit_step: int,
    new_token_string: str,
) -> dict:
    """
    Fork a new instance by replacing the token at edit_step and continuing generation.

    Retains activations 0..edit_step-1 from the original, substitutes the edited
    token, then generates the remainder with probing hooks active.
    """
    model, processor = _get_model_and_processor(model_name)

    prefix_acts  = original_result["activations"][:edit_step].tolist() if edit_step > 0 else []
    prefix_attn  = original_result["attn_weights"][:edit_step] if edit_step > 0 else []
    prefix_layer_attn = [
        layer_steps[:edit_step]
        for layer_steps in original_result.get("layer_attn_weights", [])
    ]
    prefix_ids   = original_result["token_ids"][:edit_step]    if edit_step > 0 else []

    new_ids = processor.tokenizer.encode(new_token_string, add_special_tokens=False)

    return run_inference(
        image=image,
        question=question,
        model_name=model_name,
        prefix_ids=prefix_ids + new_ids,
        prefix_activations=[np.array(a) for a in prefix_acts],
        prefix_attn=prefix_attn,
        prefix_layer_attn=prefix_layer_attn,
    )


def unload_model(model_name: str):
    """Unload a model from cache and free GPU memory."""
    import gc
    if model_name in _model_cache:
        del _model_cache[model_name]
        del _processor_cache[model_name]
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[inference] Unloaded {model_name}")


def get_loaded_model() -> str | None:
    """Return the name of the currently loaded model, or None."""
    if _model_cache:
        return next(iter(_model_cache))
    return None
