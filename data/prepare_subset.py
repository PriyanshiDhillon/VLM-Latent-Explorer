import json
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

DATASET_ID = "NOVAglow646/Monet-SFT-125K"
N_EXAMPLES = 50

OUT_DIR = Path("subset")
IMAGE_DIR = OUT_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

zip_cache = {}


def get_images_zip(dataset_name):
    if dataset_name not in zip_cache:
        zip_cache[dataset_name] = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=f"{dataset_name}/images.zip",
        )
    return zip_cache[dataset_name]


def extract_question_answer_images(messages):
    question = ""
    answer = ""
    image_refs = []

    for msg in messages:
        role = msg.get("role", "")
        for item in msg.get("content", []):
            item_type = item.get("type")

            if item_type == "image" and item.get("image"):
                image_refs.append(item["image"])

            elif item_type == "text" and item.get("text"):
                text = item["text"]

                if role == "user" and not question:
                    question = text
                elif role == "assistant":
                    answer += text

    return question, answer, image_refs


def save_image_from_zip(dataset_name, image_ref, out_path):
    zip_path = get_images_zip(dataset_name)
    member_name = Path(image_ref).name

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        if member_name not in names:
            raise FileNotFoundError(
                f"{member_name} not found in {zip_path}. "
                f"First 20 zip entries: {list(names)[:20]}"
            )

        with zf.open(member_name) as src:
            out_path.write_bytes(src.read())


def main():
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    metadata = []

    for i, ex in enumerate(ds):
        if i >= N_EXAMPLES:
            break

        example_id = f"example_{i:06d}"
        source_metadata = ex.get("metadata", {})
        dataset_name = source_metadata.get("dataset_name", "")
        messages = ex["data"]

        question, answer, image_refs = extract_question_answer_images(messages)

        local_image_paths = []
        for j, image_ref in enumerate(image_refs):
            suffix = Path(image_ref).suffix or ".jpg"
            out_image = IMAGE_DIR / f"{example_id}_{j}{suffix}"

            save_image_from_zip(dataset_name, image_ref, out_image)
            local_image_paths.append(str(out_image))

        metadata.append({
            "id": example_id,
            "source_index": i,
            "source_metadata": source_metadata,
            "dataset_name": dataset_name,
            "question": question,
            "answer": answer,
            "image_path": local_image_paths[0] if local_image_paths else None,
            "image_paths": local_image_paths,
            "image_refs": image_refs,
            "raw_data": messages,
        })

        print(f"[{i + 1}/{N_EXAMPLES}] saved {example_id} with {len(local_image_paths)} image(s)")

    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(metadata)} examples to {OUT_DIR / 'metadata.json'}")
    print(f"Saved images to {IMAGE_DIR}")


if __name__ == "__main__":
    main()