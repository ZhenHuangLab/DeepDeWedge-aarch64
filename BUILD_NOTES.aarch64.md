# DeepDeWedge (ddw) — Linux aarch64 + CUDA 13 + GB10 (sm_121)

Upstream: https://github.com/MLI-lab/DeepDeWedge

## Result: 🟢 PASS — 3 patches, none of them aarch64-specific

Pure PyTorch. The only work needed was migrating from the frozen
`pytorch-lightning==1.8.0.post1` pin (which predates torch 2.x and has no
aarch64 story) to PyTorch Lightning 2.x.

## Install

```bash
mamba create -y -p ./env -c conda-forge python=3.12
env/bin/pip install torch torchvision                 # torch 2.13.0+cu130 aarch64
env/bin/pip install "pytorch-lightning>=2.2" mrcfile pandas scikit-image \
                    scipy matplotlib pyyaml tqdm typer typer-config pexpect
env/bin/pip install -e .
```

Deliberately **ignore `requirements.txt`** — every version there is hard-pinned
to 2023/2024 releases (`pytorch-lightning==1.8.0.post1`, `mrcfile==1.5.0`,
`scipy==1.13.0`, `typer==0.16.0`, ...). Installing the pins is what makes this
package look aarch64-incompatible; it is not.

## Patch 1 — `ddw/fit_model.py`: `resume_from_checkpoint`

Lightning 2.0 moved this from `Trainer.__init__` to `Trainer.fit`. Upstream had
already written the replacement but left it commented out:

```python
    trainer = pl.Trainer(
        ...
        detect_anomaly=True,
    )                                    # drop resume_from_checkpoint=...
    ...
    trainer.fit(
        ckpt_path=resume_from_checkpoint,   # was commented out
        model=lit_unet,
        ...
    )
```

## Patch 2 — `ddw/fit_model.py`: `strategy=None`

Lightning 2.x rejects `strategy=None` with
`ValueError: You selected an invalid strategy name: strategy=None`.

```python
    strategy = pl.strategies.DDPStrategy(...) if len(devices) > 1 else "auto"
```

## Patch 3 — `ddw/utils/unet.py`: dataloader access

Lightning 2.x hands back the `DataLoader` itself instead of a `CombinedLoader`,
so `.loaders` no longer exists (`AttributeError: 'MultiEpochsDataLoader' object
has no attribute 'loaders'` at the first missing-wedge update), and
`val_dataloaders` is no longer always a list.

```python
    train_loader = getattr(self.trainer.train_dataloader, "loaders", self.trainer.train_dataloader)
    ...
    val_dls = self.trainer.val_dataloaders
    val_loader = val_dls[0] if isinstance(val_dls, (list, tuple)) else val_dls
```

## Smoke test — full three-stage pipeline

192^3 phantom (four Gaussian-smoothed spheres) with two independent noise
realisations standing in for even/odd frame half-tomograms.

| stage | command | result |
|---|---|---|
| 1 | `ddw prepare-data --config config.yaml` | 8/8 sub-tomograms selected, 6 fitting + 2 validation |
| 2 | `ddw fit-model --config config.yaml` | 3 epochs on GPU; fitting loss 1.324 -> 1.191 -> 1.089, val loss 1.212 -> 1.123 -> 1.073; missing-wedge update pass runs each 2 epochs |
| 3 | `ddw refine-tomogram --config config.yaml --model-checkpoint-file <ckpt>` | 32+32 sub-tomogram inference at ~30-40 it/s, wrote `tomo_even+tomo_odd_refined.mrc` (192^3 float32, all finite) |

Losses decrease monotonically and the refined tomogram is finite and correctly
shaped, so the denoising / missing-wedge network trains and infers correctly on
aarch64.
