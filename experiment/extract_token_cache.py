import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # Older/newer transformers builds may expose only Auto classes.
    Qwen2_5_VLForConditionalGeneration = None


ROOT = Path.home() / "VLM-Latent-Explorer"
DATA_PATH = ROOT / "data" / "subset" / "metadata.json"
OUTPUT_ROOT = ROOT / "precomputed" / "corpus_embeddings"

MODEL_PATHS = {
    "qwen": ROOT / "model" / "Qwen2.5-VL-7B-Instruct",
    "monet": ROOT / "model" / "Monet-7B",
    "lvr": ROOT / "model" / "LVR-7B",
}

LATENT_MARKERS = (
    "<abs_vis_token>",
    "</abs_vis_token>",
    "<|latent|>",
    "<latent>",
    "<visual_latent>",
)

VISION_TOKENS = (
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run offline VLM inference and cache token-level representations."
    )
    parser.add_argument(
        "--model",
        choices=["qwen", "monet", "lvr", "all"],
        default="qwen",
        help="Which local model checkpoint to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of subset examples to process.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index inside data/subset/metadata.json.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generation budget per example.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute caches that already exist.",
    )
    return parser.parse_args()


def resolve_data_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / "data" / path


def load_model(model_name):
    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing local model directory: {model_path}")

    print(f"[{model_name}] loading processor from {model_path}")
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    print(f"[{model_name}] loading model from {model_path}")
    model_classes = []
    if Qwen2_5_VLForConditionalGeneration is not None:
        model_classes.append(Qwen2_5_VLForConditionalGeneration)

    try:
        from transformers import AutoModelForVision2Seq

        model_classes.append(AutoModelForVision2Seq)
    except ImportError:
        pass

    from transformers import AutoModelForCausalLM

    model_classes.append(AutoModelForCausalLM)

    last_error = None
    model = None
    for model_cls in model_classes:
        try:
            print(f"[{model_name}] trying {model_cls.__name__}")
            model = model_cls.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True,
            )
            break
        except Exception as exc:
            last_error = exc
            print(f"[{model_name}] {model_cls.__name__} failed: {exc}")

    if model is None:
        raise RuntimeError(f"Could not load {model_name} from {model_path}") from last_error

    model.eval()
    return processor, model


def build_messages(example):
    content = []
    image_paths = example.get("image_paths") or [example.get("image_path")]
    for image_path_str in image_paths:
        if not image_path_str:
            continue
        image_path = resolve_data_path(image_path_str)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        content.append({"type": "image", "image": Image.open(image_path).convert("RGB")})

    content.append({"type": "text", "text": example["question"]})
    return [{"role": "user", "content": content}]


def tensor_to_list(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def token_text(tokenizer, token_id):
    piece = tokenizer.convert_ids_to_tokens(int(token_id))
    decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return decoded if decoded else str(piece)


def classify_tokens(tokenizer, token_ids, prompt_len):
    token_strings = []
    token_types = []
    token_sources = []

    vision_ids = {
        tokenizer.convert_tokens_to_ids(tok)
        for tok in VISION_TOKENS
        if tokenizer.convert_tokens_to_ids(tok) is not None
        and tokenizer.convert_tokens_to_ids(tok) != tokenizer.unk_token_id
    }
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    in_vision = False
    in_latent = False

    for idx, token_id in enumerate(token_ids):
        token_str = token_text(tokenizer, token_id)
        source = "prompt" if idx < prompt_len else "generated"

        if token_id == vision_start_id:
            in_vision = True
            token_type = "visual"
        elif token_id == vision_end_id:
            token_type = "visual"
            in_vision = False
        elif in_vision or token_id in vision_ids:
            token_type = "visual"
        elif any(marker in token_str for marker in LATENT_MARKERS):
            in_latent = "<abs_vis_token>" in token_str or "<|latent|>" in token_str
            token_type = "latent"
        elif in_latent and source == "generated":
            token_type = "latent"
            if "</abs_vis_token>" in token_str:
                in_latent = False
        else:
            token_type = "text"

        token_strings.append(token_str)
        token_types.append(token_type)
        token_sources.append(source)

    return token_strings, token_types, token_sources


def make_inputs(processor, messages, device):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def extend_inputs_for_full_forward(inputs, full_input_ids):
    full_inputs = dict(inputs)
    full_inputs["input_ids"] = full_input_ids
    full_inputs["attention_mask"] = torch.ones_like(full_input_ids, device=full_input_ids.device)
    return full_inputs


def extract_generated_activations(generate_hidden_states, expected_steps, hidden_size):
    if not generate_hidden_states:
        return np.empty((0, hidden_size), dtype=np.float16)

    vectors = []
    for step_hidden_states in generate_hidden_states:
        if not step_hidden_states:
            continue
        last_layer = step_hidden_states[-1]
        if last_layer.ndim != 3:
            continue
        vectors.append(last_layer[0, -1, :].detach().float().cpu().numpy())

    if not vectors:
        return np.empty((0, hidden_size), dtype=np.float16)

    activations = np.stack(vectors).astype(np.float16)
    if len(activations) > expected_steps:
        activations = activations[:expected_steps]
    return activations


def run_example(model_name, processor, model, example, max_new_tokens):
    messages = build_messages(example)
    inputs = make_inputs(processor, messages, model.device)
    prompt_len = int(inputs["input_ids"].shape[1])

    with torch.no_grad():
        prompt_outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    prompt_activations = (
        prompt_outputs.hidden_states[-1][0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float16)
    )

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    full_input_ids = generated.sequences
    generated_ids = full_input_ids[:, prompt_len:]
    generated_activations = extract_generated_activations(
        generated.hidden_states,
        expected_steps=int(generated_ids.shape[1]),
        hidden_size=int(prompt_activations.shape[1]),
    )

    if len(generated_activations) < int(generated_ids.shape[1]):
        missing = int(generated_ids.shape[1]) - len(generated_activations)
        filler = np.full((missing, prompt_activations.shape[1]), np.nan, dtype=np.float16)
        generated_activations = np.concatenate([generated_activations, filler], axis=0)

    activations = np.concatenate([prompt_activations, generated_activations], axis=0)
    all_token_ids = full_input_ids[0].detach().cpu().tolist()
    generated_token_ids = generated_ids[0].detach().cpu().tolist()
    token_strings, token_types, token_sources = classify_tokens(
        processor.tokenizer,
        all_token_ids,
        prompt_len,
    )
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return {
        "example_id": example["id"],
        "model_name": model_name,
        "activations": activations,
        "token_ids": np.array(all_token_ids, dtype=np.int64),
        "generated_token_ids": np.array(generated_token_ids, dtype=np.int64),
        "token_strings": np.array(token_strings, dtype=object),
        "token_types": np.array(token_types, dtype=object),
        "token_sources": np.array(token_sources, dtype=object),
        "prompt_len": np.array(prompt_len, dtype=np.int64),
        "generated_text": np.array(generated_text, dtype=object),
        "question": np.array(example.get("question", ""), dtype=object),
        "answer": np.array(example.get("answer", ""), dtype=object),
        "image_path": np.array(example.get("image_path", ""), dtype=object),
        "image_paths": np.array(example.get("image_paths", []), dtype=object),
        "source_metadata": np.array(json.dumps(example.get("source_metadata", {})), dtype=object),
    }


def save_cache(model_name, example_id, result):
    out_dir = OUTPUT_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{example_id}.npz"
    np.savez_compressed(out_path, **result)
    return out_path


def run_model(model_name, examples, args):
    processor, model = load_model(model_name)
    selected = examples[args.start : args.start + args.limit]

    for offset, example in enumerate(selected, start=args.start):
        out_path = OUTPUT_ROOT / model_name / f"{example['id']}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[{model_name}] skip existing {out_path}")
            continue

        print(f"[{model_name}] processing {offset}: {example['id']}")
        result = run_example(
            model_name=model_name,
            processor=processor,
            model=model,
            example=example,
            max_new_tokens=args.max_new_tokens,
        )
        saved_path = save_cache(model_name, example["id"], result)
        print(
            f"[{model_name}] saved {saved_path} "
            f"tokens={len(result['token_ids'])} "
            f"generated={len(result['generated_token_ids'])}"
        )

    del model
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    examples = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    model_names = ["qwen", "monet", "lvr"] if args.model == "all" else [args.model]

    print(f"Loaded {len(examples)} examples from {DATA_PATH}")
    print(f"Running models: {model_names}")
    print(f"Range: start={args.start}, limit={args.limit}")

    for model_name in model_names:
        run_model(model_name, examples, args)

    print("Token cache extraction complete.")


if __name__ == "__main__":
    main()
