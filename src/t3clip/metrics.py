import warnings

import torch
from torch.nn import functional as F

from t3clip.engine import autocast


def retrieval_metrics(
    query_features, gallery_features, query_ids, gallery_ids, chunk_size=256
):
    if len(gallery_ids) == 0:
        raise ValueError("empty gallery")
    keep = torch.isin(query_ids, gallery_ids)
    if not keep.any():
        raise ValueError("no queries have a gallery match")
    if not keep.all():
        warnings.warn("queries without gallery matches were excluded", stacklevel=2)
    queries, ids = query_features[keep].float(), query_ids[keep]
    gallery = gallery_features.float()
    totals = {"t2i_r1": 0.0, "t2i_r5": 0.0, "t2i_r10": 0.0, "t2i_map": 0.0}
    ranks = torch.arange(1, len(gallery_ids) + 1, dtype=torch.float32)
    for start in range(0, len(ids), chunk_size):
        ranking = (queries[start : start + chunk_size] @ gallery.T).argsort(
            dim=1, descending=True
        )
        matches = gallery_ids[ranking].eq(ids[start : start + chunk_size, None])
        for k in (1, 5, 10):
            totals[f"t2i_r{k}"] += matches[:, :k].any(dim=1).float().sum().item()
        ap = ((matches.float().cumsum(dim=1) / ranks) * matches).sum(
            dim=1
        ) / matches.sum(dim=1)
        totals["t2i_map"] += ap.sum().item()
    return {key: value / len(ids) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, image_loader, text_loader, device, precision):
    previous = model.training
    model.eval()
    features, identities = [], []
    try:
        for loader, key, encoder in (
            (image_loader, "images", model.encode_image),
            (text_loader, "text_tokens", model.encode_text),
        ):
            encoded, ids = [], []
            for batch in loader:
                with autocast(device, precision):
                    values = F.normalize(
                        encoder(batch[key].to(device, non_blocking=True)), dim=-1
                    )
                encoded.append(values.float().cpu())
                ids.append(batch["person_ids"].cpu())
            if not encoded:
                raise ValueError("empty evaluation split")
            features.append(torch.cat(encoded))
            identities.append(torch.cat(ids))
        return retrieval_metrics(features[1], features[0], identities[1], identities[0])
    finally:
        model.train(previous)
