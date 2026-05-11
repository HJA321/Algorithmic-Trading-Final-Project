# Columbia IEOR 4733: Algorithmic Trading Final Project

**Team Members:** Jiaan Han, Xiangpeng Huang, Tianle Liu

This repository contains the code for the IEOR E4733 Algorithmic Trading final project:
a PyTorch Attn-LOB + reinforcement-learning market-making system with baseline,
latency/order-loss, and regime-sensitivity evaluations.

Start with:

- `Our-Code/README.md` for the main code overview
- `Our-Code/SYSTEM_README.md` for the API/CLI workflow
- `Our-Code/PROJECT_RUNBOOK.md` for the reproduction checklist
- `PROJECT_REPORT.tex` for the written project report source
- `PROJECT_REPORT.pdf` for the compiled project report

Generated model artifacts and figures are written under `Our-Code/experiment_results/`
and are intentionally ignored by git.

## Submission Scope

For the code submission, `Our-Code/` is the self-contained implementation folder.
It includes the data preparation scripts, Attn-LOB pretraining code, RL market-making
environment, baselines, latency/regime evaluation scripts, and clear API-based
entry points.

Large raw data, generated processed data, model checkpoints, plots, and local
virtual environments should not be submitted. They are ignored through `.gitignore`
and can be regenerated using the commands documented in `Our-Code/SYSTEM_README.md`.

If data must be included in the GitHub repository, prefer processed data only
(`Our-Code/data/sz_l2_processed/` and
`lobster_attnlob_us_repro/data/lobster_processed/`) rather than raw archives or
full model checkpoints.
