# presto-benchmarking
[`presto`](https://github.com/cole-group/presto) is a tool for fitting bespoke SMIRNOFF force fields for your molecule(s) of interest. This repo contains a `Snakemake` workflow that reproduces all of the results in the `presto` paper, with the exception of the Tango relative binding free energy (RBFE) calculations. Tango is not publicly available, so its precomputed predictions are committed under `benchmarking/rbfe_sandbox/` and the workflow only re-runs the downstream analysis.

To rerun the full workflow (the default target builds results for every section of the paper), [install `pixi`](https://pixi.prefix.dev/latest/installation/) and run:
```bash
git clone https://github.com/fjclark/presto-benchmarking.git
cd presto-benchmarking
pixi install --all
pixi run snakemake --cores all
```

## Reproducing the paper

The default target (`rule all`) builds results for every section of the paper. To rebuild a single section instead of everything, pass its target explicitly:
```bash
pixi run snakemake --cores all benchmarking/tnet500/analysis/test/default/metrics.json
```

| Paper section | Snakemake target |
| --- | --- |
| TorsionNet500 test set | `benchmarking/tnet500/analysis/test/default/metrics.json` |
| Workflow component ablations (on TorsionNet500) | `benchmarking/tnet500/analysis/validation/ablations/metrics.json` |
| JACS fragment torsion scans | `benchmarking/jacs_fragments/analysis/test/default/metrics.json` |
| Folmsbee relative conformer energies | `benchmarking/folmsbee_conformers/analysis/test/aimnet2/aggregate_stats.csv` |
| Congeneric series type specificity | `benchmarking/tyk2_congeneric_series/analysis/retrain_error_summary.csv` |
| Relative binding free energies | `benchmarking/rbfe_sandbox/results/bootstrap_statistics.csv` |
| **SI** Dataset descriptor statistics | `benchmarking/analysis/smiles_descriptors/smiles_descriptor_aggregate_mean_std.{csv,tex}` |
| **SI** presto validation per-atom energy RMSEs | `benchmarking/analysis/presto_fit_validation/presto_fit_validation_error_aggregate.{csv,tex}` |
| **SI** TorsionNet500 with B3LYP-D3(BJ)/DZVP reference | `benchmarking/tnet500_reopt_v4/analysis/test/default/metrics.json` and `.../validation/ablations/metrics.json` |
| **SI** Fit reproducibility / parameter convergence (TYK2) | `benchmarking/tyk2_reproducibility/analysis/parameter_variability/offxml_variability_summary.tex` |
| **SI** Folmsbee phosphate/sulfonamide rerun (no MSM/MLP-min) | `benchmarking/folmsbee_conformers/analysis/phosphate_sulphonamide/aimnet2_no_msm_no_min/aggregate_stats.csv` |
| **SI** TYK2 cyclopropanecarboxamide edge torsion scans | `benchmarking/tyk2_cyclopropyl_edges_torsions/analysis/metrics.json` |

### Requirements and notes

- **GPU.** The `presto` fits (`run_presto`) and the MLP-based analyses require a GPU and CUDA. Most fitting/analysis rules request a GPU via their SLURM resources.
- **Committed outputs.** Most of `benchmarking/` is git-ignored, but the results needed to inspect the paper's findings without rerunning anything are committed:
  - the **combined bespoke force fields** (`combined_force_field.offxml`) for each paper dataset, under `benchmarking/<dataset>/output/...`. Note these are large — the Folmsbee and JACS force fields in particular.
  - the **lightweight analysis outputs** that back the tables and figures: the torsion-scan `metrics.json` files, Folmsbee `aggregate_stats.csv`, the descriptor and validation-error aggregates, the congeneric/reproducibility summaries, and the RBFE `bootstrap_statistics.csv` / `statistical_tests.csv`.

  Heavy or intermediate artifacts (per-molecule fits, `minimized.json`, `*.sqlite` torsion databases, plots/HTML, and all exploratory/"saved" runs) remain ignored. The exact re-included paths are listed in `.gitignore`.
- **Network downloads.** Benchmark *inputs* are not committed and are downloaded at runtime from QCArchive and from the Folmsbee/Hutchison `conformer-benchmark` GitHub repository.
- **Pixi environments** are selected per rule:
  - `presto051` is auto-selected (via `get_pixi_env`) for the `tnet500` and `jacs_fragments` datasets, which were run with `presto` 0.5.1; everything else uses `default` (`presto` 0.6.0).
  - The Folmsbee set uses the `aimnet2` config (AIMNet2 as the reference MLP), matching the paper.
  - The JACS full-molecule-fit and TYK2 cyclopropyl torsion analyses use the `no-openeye` environment, matching the environment used by SandboxAQ.
- **RBFE.** The free-energy section runs entirely from the committed data in `benchmarking/rbfe_sandbox/`; no GPU or download is needed for it.

## Running on a SLURM cluster

Create a workflow-specific profile (the `profiles/` directory is git-ignored):
```bash
mkdir -p profiles/default
cat > profiles/default/config.yaml << 'EOF'
executor: slurm
jobs: 100
default-resources:
  mem_mb: 4000
  runtime: 60          # minutes
  slurm_partition: ""  # set your partition, e.g. "gpu"
  slurm_account: ""    # set your account if required
latency-wait: 60
rerun-incomplete: true
EOF
```

Snakemake automatically picks up `profiles/default` as the default profile, so you can simply run:
```bash
pixi run snakemake
```

### Changing the queue/partition for the fitting jobs

The GPU fitting rules (`run_presto`, and likewise `run_tyk2_reproducibility`, `run_tyk2_congeneric_series`, `run_tyk2_congeneric_series_retrain`, `run_1mer_backbone_joint_fit`) hard-code their SLURM partition and GPU request in a `resources:` block, e.g. in the `Snakefile`:
```python
rule run_presto:
    ...
    resources:
        mem_mb=8000,
        runtime=120,             # minutes
        slurm_partition="gpu-s_free",   # <- change to your GPU queue
        slurm_extra="--gpus-per-task=1" # <- adjust GPU request if needed
```
Because a value set on the rule overrides the profile's `default-resources`, to switch queue you can edit the `resources:` block(s) in the `Snakefile` directly (change `slurm_partition`, and `slurm_extra` for the GPU/account flags).
