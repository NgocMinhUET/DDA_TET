# DDA-TET v0.1: Diffusion-Distilled Adaptive Task Enhancement Tiles

This repository contains a clean proof-of-concept implementation of **DDA-TET v0.1**:

> Offline diffusion restoration teacher, online adaptive tile student for machine-oriented post-processing in VCM.

The online student is intentionally ultra-lightweight:

\[
x_m = x_c + \sum_{k=1}^{K} \alpha_k \mathcal{E}(T_k)
\]

where `T_k` are RGB-domain basis tiles, `alpha_k` are predicted by a tiny MLP from global statistics, and `E(.)` repeats a tile over the full image.

## What is included

- Core DDA-TET student modules.
- Statistics normalization.
- Distillation, sparsity, and orthogonality losses.
- Optional FPN feature loss hooks for Faster R-CNN-FPN.
- Dataset cache reader for teacher residual distillation.
- VTM encode/decode command wrapper.
- Forward tests and tiny synthetic training smoke test.

## Quick test

```bash
pip install -r requirements.txt
python -m tests.test_forward_dda_tet
python scripts/smoke_train_synthetic.py
```

## Expected cache format

The real training script expects a JSONL index. Each line points to tensors or images:

```json
{"xc":"sample_xc.pt", "rd":"sample_rd.pt", "stats":[37, 45000, 0.45, 0.18, 0.12, 5.7, 0.22]}
```

Tensor files should store CHW tensors in `[0,1]`.

## Notes

- Diffusion teacher is offline only. This repository provides the interface/wrapper, not pretrained DiffBIR weights.
- VTM is not bundled. Use `data/encode_vtm.py` with your local VTM binaries.
- Start with core distillation loss only; enable FPN/task losses after the student learns a stable residual.
# DDA_TET
