import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "model" / "Qwen2.5-VL-7B-Instruct"
DATA_PATH = ROOT / "data" / "subset" / "metadata.json"


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / "data" / path


def main():
    print("Loading metadata...")
    items = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    ex = items[0]

    image_path = resolve_path(ex["image_path"])
    question = ex["question"]
    ground_truth = ex.get("answer", "")

    print(f"Using example: {ex['id']}")
    print(f"Image path: {image_path}")
    print(f"Model path: {MODEL_PATH}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

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
    ).to(model.device)

    print("Running generation...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\n================ QUESTION ================")
    print(question)

    print("\n================ MODEL OUTPUT ================")
    print(output_text)

    print("\n================ GROUND TRUTH ================")
    print(ground_truth[:1500])

    print("\nSmoke test completed successfully.")


if __name__ == "__main__":
    main()
