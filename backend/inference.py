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
_model_cache: dict = {}
_processor_cache: dict = {}

MODEL_IDS = {
    "qwen":  "/gpfs/scratch1/shared/kguanmma/Qwen2.5-VL-7B-Instruct",
    "monet": "/gpfs/scratch1/shared/kguanmma/Monet-7B",
    "lvr":   "/gpfs/scratch1/shared/kguanmma/LVR-7B",
}

# Token type classification
LATENT_TRIGGER_TOKEN = "<|latent|>"   # adjust to whatever Monet/LVR actually use


def _get_model_and_processor(model_name: str):
    """Load (and cache) a model + processor by name."""
    if model_name not in _model_cache:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        model_id = MODEL_IDS[model_name]
        print(f"[inference] Loading {model_id} ...")

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model.eval()
        _model_cache[model_name] = model
        _processor_cache[model_name] = processor

    return _model_cache[model_name], _processor_cache[model_name]


# ---------------------------------------------------------------------------
# Hook management
# ---------------------------------------------------------------------------

class ActivationStore:
    """Collects activations and attention weights injected by forward hooks."""

    def __init__(self):
        self.hidden_states: list[np.ndarray] = [] 
        self.attn_weights: list[np.ndarray] = []  
        self._hooks: list = []

    def clear(self):
        self.hidden_states.clear()
        self.attn_weights.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _register_hooks(model, store: ActivationStore, target_layer: int = -1):
    """
    Register forward hooks on the last (or specified) transformer decoder layer.

    - Hidden state hook: captures the output of the target layer.
    - Attention hook: captures cross-attention (or self-attention) weights.
    """
    layers = model.model.language_model.layers
    layer = layers[target_layer]

    def _hidden_hook(module, input, output):
        if store._skip_next:
            store._skip_next = False
            return
        hs = output[0][-1, :].detach().float().cpu().numpy()  # ← remove the [0, ] batch dim
        store.hidden_states.append(hs)

    def _attn_hook(module, input, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            w = output[1][0, :, -1, :].detach().float().cpu().numpy()
            store.attn_weights.append(w)
        else:
            store.attn_weights.append(None)

    store._hooks.append(layer.register_forward_hook(_hidden_hook))
    if hasattr(layer, "self_attn"):
        store._hooks.append(layer.self_attn.register_forward_hook(_attn_hook))


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
    Label each generated token as 'text', 'visual', or 'latent'.

    - visual  : image patch tokens inserted by the processor
    - latent  : tokens generated after a latent trigger (Monet/LVR only)
    - text    : everything else
    """
    vocab = processor.tokenizer.get_vocab()
    latent_id = vocab.get(LATENT_TRIGGER_TOKEN, None)

    vision_start_id = vocab.get("<|vision_start|>", None)
    vision_end_id   = vocab.get("<|vision_end|>", None)

    types = []
    in_visual = False
    in_latent = False

    for tid in token_ids:
        if tid == vision_start_id:
            in_visual = True
            types.append("visual")
        elif tid == vision_end_id:
            in_visual = False
            types.append("visual")
        elif in_visual:
            types.append("visual")
        elif model_name in ("monet", "lvr") and tid == latent_id:
            in_latent = True
            types.append("latent")
        elif in_latent:
            decoded = processor.tokenizer.decode([tid])
            if decoded.strip() in ("", "\n") or tid == processor.tokenizer.eos_token_id:
                in_latent = False
                types.append("text")
            else:
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
    prefix_ids: Optional[list[int]] = None,
    prefix_activations: Optional[list[np.ndarray]] = None,
    prefix_attn: Optional[list[np.ndarray]] = None,
) -> dict:
    """
    Run a forward pass and return all extracted data needed by the dashboard.

    Parameters
    ----------
    image           : PIL Image
    question        : user question string
    model_name      : 'qwen' | 'monet' | 'lvr'
    max_new_tokens  : generation budget
    prefix_ids      : if set, prepend these token ids (for edited-instance continuation)
    prefix_activations : activations from the prefix (editing case)
    prefix_attn        : attention weights from the prefix (editing case)

    Returns
    -------
    dict with keys:
        activations    : np.ndarray (T, D)
        attn_weights   : list[np.ndarray | None]  length T, each (num_heads, src_len) or None
        token_types    : list[str]  length T
        token_strings  : list[str]  length T
        token_ids      : list[int]  length T
        generated_text : str
        image_grid_hw  : tuple[int, int]  — spatial grid dims of image patches
    """
    model, processor = _get_model_and_processor(model_name)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    img_w, img_h = image.size
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        _, grid_h, grid_w = image_grid_thw[0].tolist()
    else:
        grid_h = img_h // 14
        grid_w = img_w // 14

    store = ActivationStore()
    store._skip_next = True
    _register_hooks(model, store)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_attentions=True, 
        )

    store.remove_hooks()

    input_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0, input_len:].tolist()

    token_strings = [
        processor.tokenizer.decode([tid], skip_special_tokens=False)
        for tid in gen_ids
    ]
    generated_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

    all_activations   = (prefix_activations or []) + store.hidden_states
    all_attn          = (prefix_attn or [])        + store.attn_weights
    all_ids           = (prefix_ids or [])         + gen_ids

    token_types = _classify_token_types(all_ids, processor, grid_h * grid_w, model_name)

    activations_array = np.stack(all_activations) if all_activations else np.empty((0, model.config.hidden_size))

    return {
        "activations":    activations_array,
        "attn_weights":   all_attn,
        "token_types":    token_types,
        "token_strings":  token_strings,
        "token_ids":      all_ids,
        "generated_text": generated_text,
        "image_grid_hw":  (grid_h, grid_w),
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
    prefix_ids   = original_result["token_ids"][:edit_step]    if edit_step > 0 else []

    new_ids = processor.tokenizer.encode(new_token_string, add_special_tokens=False)

    return run_inference(
        image=image,
        question=question,
        model_name=model_name,
        prefix_ids=prefix_ids + new_ids,
        prefix_activations=[np.array(a) for a in prefix_acts],
        prefix_attn=prefix_attn,
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