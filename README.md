# 3T-CLIP

Improving Time-to-Target Performance in Text-Based Person Search

Jiwoong Choi, Hyunjoo Song

3T-CLIP fine-tunes CLIP ViT-B/16 with automatic mixed precision (AMP), an
exponential moving average (EMA), and gradient checkpointing. This
architecture-preserving training strategy adds no retrieval-specific modules
or inference-time computation and is evaluated by time-to-target retrieval
accuracy and peak GPU memory on CUHK-PEDES, ICFG-PEDES, and RSTPReid.

## Requirements / Installation

Python 3.12, uv, and a compatible NVIDIA CUDA GPU and driver are required.

```sh
uv sync --extra cuda
```

## Dataset preparation

Dataset images, annotations, pretrained weights, and other training resources
are not provided with this release. Researchers reproducing the results must
obtain the required resources under their respective access and license terms
and place them in the expected locations. Datasets are not auto-downloaded.
OpenCLIP downloads the default `openai` initialization when needed; use
`cache_dir` in the YAML config for its cache. If weights cannot be auto-downloaded,
obtain and place them yourself and set `pretrained` to their local path when training.

The `data_root` setting names the parent of the following directories:

```text
datasets/
  CUHK-PEDES/
    reid_raw.json
    imgs/
  ICFG-PEDES/
    ICFG-PEDES.json
    imgs/
  RSTPReid/
    data_captions.json
    imgs/
```

## Training

All settings are read from YAML; `--config` is the only command-line
argument and takes one dataset config. `configs/config.yaml` holds every
default and is loaded automatically first; the dataset config overrides it.

Set `data_root` in `configs/config.yaml` to the parent directory shown above,
not an individual dataset directory. Each dataset config sets `output_dir` for
checkpoints and metrics and `checkpoint` for evaluation; edit them as needed.
Set `cache_dir` if you want a custom model cache.

```sh
uv run --extra cuda t3clip-train --config configs/cuhk_pedes.yaml
uv run --extra cuda t3clip-train --config configs/icfg_pedes.yaml
uv run --extra cuda t3clip-train --config configs/rstpreid.yaml
```

For ICFG-PEDES or RSTPReid, set `fold_index` to 0, 1, 2, 3, or 4 in the
matching dataset config and give each fold a separate `output_dir`. Leave
`fold_index: null` for CUHK-PEDES.

## Evaluation

Evaluate selected weights on the official test split, never the holdout.
`checkpoint` in the dataset config names the weights, and `test_metrics.json`
is written to its `output_dir`:

```sh
uv run --extra cuda t3clip-eval --config configs/cuhk_pedes.yaml
```

For the other datasets, use the matching dataset config and set `checkpoint`
to the `best.pt` from the fold with the highest validation `t2i_r1`, never
selected by test accuracy.
CUHK-PEDES uses its native validation split; ICFG-PEDES and RSTPReid use
identity-disjoint five-fold splits of the official training set. Their reported
official-test results evaluate the best-validation fold model, not a model
retrained on the full training set. Results are written to `test_metrics.json`
as fractions. Dataset, architecture, and image dimensions must match the
checkpoint. Only load trusted checkpoints.

## Results

Our AMP+EMA+GC configuration on the official test splits; retrieval metrics
are percentages. T_rho is cumulative training time to the validation R@1
target, not time to the best checkpoint or to an official-test target.
Peak memory is training-phase peak allocated GPU memory on an RTX 5070 Ti,
excluding evaluation.

| Dataset | R@1 | R@5 | R@10 | mAP | rho (%) | T_rho (h) | Peak memory (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CUHK-PEDES | 70.11 | 87.90 | 92.72 | 63.93 | 68.19 | 0.37 | 3.47 |
| ICFG-PEDES | 59.55 | 77.28 | 83.20 | 36.75 | 56.74 | — | 3.47 |
| RSTPReid | 53.35 | 77.45 | 84.75 | 44.66 | 54.05 | 0.38 | 3.47 |

The manuscript does not state an exact ICFG-PEDES T_rho. The RSTPReid time
is for the best-validation fold; only one of five folds reaches the target.
Secondary-dataset memory was measured on the first fold only.

## Reproducing time-to-target

`train_gpu_hours` measures cumulative training-loop time, including data loading,
updates, and step logging, but excluding validation, checkpoint writes, and
setup. CUDA synchronizes at training-loop boundaries. `wall_hours` sums epoch
intervals including validation and EMA swaps, excluding setup and checkpoint
writes.
Read `validation.jsonl`: T_rho is `train_gpu_hours` at the earliest epoch
reaching `t2i_r1` of 0.6819 (CUHK-PEDES), 0.5674 (ICFG-PEDES), or 0.5405
(RSTPReid). If no epoch reaches the target, report it as not reached.
Local `wall_hours` is not the paper's end-to-end runtime measurement.

For peak-memory measurement, set `max_steps: 400` in the matching YAML recipe
and `fold_index: 0` for five-fold datasets. Divide the maximum step-log
`peak_allocated` by 1024³ for GiB. This counts successful updates, whereas the
paper counted training batches: these coincide only with accumulation 1 and
no skipped AMP updates. Use a separate output directory for this measurement
and restore `max_steps: 0` for full training.

## Acknowledgement

This release uses CLIP through OpenCLIP. We acknowledge IRRA for the published
plain-CLIP fine-tuning reference used to define the time-to-target thresholds
and for training practices adopted in the paper's protocol.

## Citation

⚠️ The paper is currently under review. The citation below will be updated once the manuscript is accepted and assigned final publication details.

```bibtex
@article{choi2026timetotarget,
  title = {Improving Time-to-Target Performance in Text-Based Person Search},
  author = {Choi, Jiwoong and Song, Hyunjoo},
  year = {2026},
  note = {Manuscript in preparation}
}
```

## License

MIT. See `LICENSE`. Dataset access terms and pretrained-model licenses remain
separate from this implementation's license.
