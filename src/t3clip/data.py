import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
DATASETS = {
    "cuhk-pedes": ("CUHK-PEDES", "reid_raw.json", "file_path"),
    "icfg-pedes": ("ICFG-PEDES", "ICFG-PEDES.json", "file_path"),
    "rstpreid": ("RSTPReid", "data_captions.json", "img_path"),
}


@dataclass(frozen=True)
class Record:
    image_path: Path
    person_id: int
    captions: tuple[str, ...]
    split: str


def load_records(dataset, data_root):
    directory, annotation, image_key = DATASETS[dataset]
    root = Path(data_root).expanduser() / directory
    payload = json.loads((root / annotation).read_text())
    if not isinstance(payload, list):
        raise ValueError("annotations must be a list of objects")
    records = []
    for row in payload:
        captions = tuple(
            str(text) for text in row.get("captions", ()) if str(text).strip()
        )
        image = root / "imgs" / row[image_key]
        if captions and image.is_file():
            records.append(Record(image, int(row["id"]), captions, str(row["split"])))
    if not records:
        raise ValueError("no annotated images found")
    return records


def split_records(records, dataset, fold_index=None):
    train = [r for r in records if r.split == "train"]
    if dataset == "cuhk-pedes":
        if fold_index is not None:
            raise ValueError("native validation does not use a fold")
        return train, [r for r in records if r.split == "val"]
    if type(fold_index) is not int or not 0 <= fold_index < 5:
        raise ValueError("fold_index must be an integer from 0 through 4")
    identities = sorted({r.person_id for r in train})
    random.Random(42).shuffle(identities)
    holdout = set(identities[fold_index::5])
    return [r for r in train if r.person_id not in holdout], [
        r for r in train if r.person_id in holdout
    ]


def build_transform(
    height=384, width=128, training=False, img_aug=True, img_hflip=True
):
    operations = [T.Resize((height, width), interpolation=T.InterpolationMode.BILINEAR)]
    if training:
        if img_hflip:
            operations.append(T.RandomHorizontalFlip(0.5))
        if img_aug:
            operations.extend([T.Pad(10), T.RandomCrop((height, width))])
    operations.extend([T.ToTensor(), T.Normalize(MEAN, STD)])
    if training and img_aug:
        operations.append(
            T.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=MEAN)
        )
    return T.Compose(operations)


class PersonDataset(Dataset):
    def __init__(self, records, transform=None, modality="pairs"):
        self.transform = transform
        self.modality = modality
        self.items = (
            [(r, text) for r in records for text in r.captions]
            if modality != "images"
            else [(r, None) for r in records]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        record, text = self.items[index]
        item = {"person_id": record.person_id, "image_path": str(record.image_path)}
        if self.modality != "texts":
            with Image.open(record.image_path) as image:
                item["image"] = self.transform(image.convert("RGB"))
        if text is not None:
            item["text"] = text
        return item


def collate(items, tokenizer):
    batch = {
        "person_ids": torch.tensor([i["person_id"] for i in items]),
        "image_paths": [i["image_path"] for i in items],
    }
    if "image" in items[0]:
        batch["images"] = torch.stack([i["image"] for i in items])
    if "text" in items[0]:
        batch["text_tokens"] = tokenizer([i["text"] for i in items])
    return batch


def make_loader(
    records, tokenizer, transform, batch_size, num_workers, device, modality="pairs"
):
    training = modality == "pairs"
    return DataLoader(
        PersonDataset(records, transform, modality),
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=partial(collate, tokenizer=tokenizer),
    )


def make_eval_loaders(records, tokenizer, transform, batch_size, num_workers, device):
    return tuple(
        make_loader(
            records, tokenizer, transform, batch_size, num_workers, device, modality
        )
        for modality in ("images", "texts")
    )
