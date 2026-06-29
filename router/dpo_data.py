"""
dpo_data.py — POPE data loading, train/valid split, preference pair construction.

Splits the 500 COCO val2014 images 8:2 by image (to prevent leakage).
Loads POPE questions, filters to train/valid sets.
"""
import json
import os
import random
from typing import List, Tuple

POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"


def load_pope_questions() -> List[dict]:
    """Load all 9000 POPE questions from the 3 subsets."""
    all_questions = []
    for subset in ["random", "popular", "adversarial"]:
        path = os.path.join(POPE_DIR, f"coco_pope_{subset}.json")
        for line in open(path, "r", encoding="utf-8"):
            q = json.loads(line)
            q["subset"] = subset
            all_questions.append(q)
    return all_questions


def split_by_image(questions: List[dict], train_ratio=0.8, seed=42):
    """
    Split questions by unique image, so no image appears in both train and valid.
    Returns (train_questions, valid_questions).
    """
    rng = random.Random(seed)

    # Group by image
    img_to_qs = {}
    for q in questions:
        img_to_qs.setdefault(q["image"], []).append(q)

    images = sorted(img_to_qs.keys())
    rng.shuffle(images)

    n_train = int(len(images) * train_ratio)
    train_images = set(images[:n_train])
    valid_images = set(images[n_train:])

    train_qs = [q for q in questions if q["image"] in train_images]
    valid_qs = [q for q in questions if q["image"] in valid_images]

    print(f"Split: {len(train_images)} train images ({len(train_qs)} questions), "
          f"{len(valid_images)} valid images ({len(valid_qs)} questions)")
    return train_qs, valid_qs


def make_question_batches(questions: List[dict], batch_size=1):
    """Yield batches of questions. batch_size=1 for per-sample DPO."""
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        yield batch


class POPEDataset:
    """Lightweight dataset for DPO training on POPE."""

    def __init__(self, questions: List[dict]):
        self.questions = questions

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        image_path = os.path.join(IMAGE_DIR, q["image"])
        return {
            "image_path": image_path,
            "question": q["text"],
            "label": q["label"],        # "yes" or "no"
            "question_id": q.get("question_id", 0),
            "subset": q.get("subset", ""),
        }
