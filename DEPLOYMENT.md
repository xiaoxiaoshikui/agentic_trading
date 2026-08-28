# Deployment Plan: Vast.ai + Hugging Face Hub

This repository is already public on GitHub. The next deployment step should separate compute from artifact hosting:

- **Vast.ai**: rent short-lived GPU instances for deep multimodal reruns.
- **Hugging Face Hub**: publish curated public artifacts such as result tables, selected paper source, architecture figures, and configs.

Do not use either platform to publish raw market/news data, real API keys, `.env`, private reviews, or full local caches unless their license and privacy status are clear.

## Current Local Status

- GitHub remote is configured: `https://github.com/xiaoxiaoshikui/agentic_trading.git`.
- Hugging Face CLI exists inside the project virtualenv: `.venv/bin/hf`.
- Hugging Face CLI is not currently logged in. Run `.venv/bin/hf auth login`.
- Vast.ai CLI can be installed in the project virtualenv with `pip install vastai`; use `.venv/bin/vastai` if it is not on `PATH`.
- The local workspace is about 9 GB because it includes data, caches, and generated experiment outputs. The public GitHub repo intentionally excludes those large artifacts.

## Install and Authenticate CLIs

Vast.ai official CLI install options:

```bash
curl -fsSL https://vast.ai/install.sh | bash
vastai set api-key YOUR_VAST_API_KEY
vastai show user
```

If you prefer project-local Python installation:

```bash
source .venv/bin/activate
pip install -r requirements-deploy.txt
vastai set api-key YOUR_VAST_API_KEY
```

Hugging Face:

```bash
source .venv/bin/activate
hf auth login
hf auth whoami
```

## Vast.ai GPU Workflow

### SSH permission deadlock

If a Vast instance logs this error:

```text
Authentication refused: bad ownership or modes for file /root/.ssh/authorized_keys
```

the failure is happening inside `sshd` before public-key authentication. The key can be correct and still be refused because OpenSSH requires safe ownership and permissions on `/root`, `/root/.ssh`, and `/root/.ssh/authorized_keys`.

Use the standalone repair script through Vast's on-start command:

```bash
curl -fsSL https://raw.githubusercontent.com/xiaoxiaoshikui/agentic_trading/main/deploy/vast_ssh_repair.sh | bash
```

For an instance that also bootstraps this project, the main bootstrap script already runs the same repair loop during startup:

```bash
RUN_MODE=smoke bash -lc 'curl -fsSL https://raw.githubusercontent.com/xiaoxiaoshikui/agentic_trading/main/deploy/vast_bootstrap.sh | bash'
```

The repair loop repeatedly applies:

```bash
chown root:root /root /root/.ssh /root/.ssh/authorized_keys
chmod go-w /root
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
```

This is an on-start mitigation, so it does not require SSH access to be working first.

### Controlled image test

To determine whether the latest vLLM-pinned image caused the SSH failure or whether the issue is host-specific, run four small smoke launches:

| Image | On-start repair | Expected interpretation |
| --- | --- | --- |
| Last known good image | off | Baseline. If this fails, suspect host/platform/key injection. |
| Last known good image | on | Should pass unless host setup is severely broken. |
| New vLLM-pinned image | off | If only this fails, suspect image-level ownership/mode changes. |
| New vLLM-pinned image | on | If this passes, the image is usable with startup repair. |

Prefer verified hosts with working direct SSH ports, avoid hosts that recently stalled on image pulls, and destroy failed smoke instances quickly.

Search for one verified RTX 4090 instance with direct SSH:

```bash
.venv/bin/vastai search offers 'gpu_name=RTX_4090 num_gpus=1 verified=true direct_port_count>=1 rentable=true' -o 'dlperf_usd-'
```

Create a smoke instance. Replace `OFFER_ID` with the selected offer.

```bash
.venv/bin/vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
  --disk 80 \
  --onstart-cmd "RUN_MODE=smoke bash -lc 'curl -fsSL https://raw.githubusercontent.com/xiaoxiaoshikui/agentic_trading/main/deploy/vast_bootstrap.sh | bash'" \
  --ssh \
  --direct
```

The bootstrap script clones the repository, creates `.venv`, installs `requirements.txt` plus `requirements-research.txt`, and runs the lightweight unit checks.

After the instance is running:

```bash
.venv/bin/vastai show instance INSTANCE_ID
.venv/bin/vastai ssh-url INSTANCE_ID
```

Copy the local data needed for the main ICAIF real-news experiment:

```bash
.venv/bin/vastai copy local:data/multimodal_history_real/ C.INSTANCE_ID:/workspace/agentic_trading/data/multimodal_history_real/
.venv/bin/vastai copy local:experiments/data_cache/ C.INSTANCE_ID:/workspace/agentic_trading/experiments/data_cache/
.venv/bin/vastai copy local:experiments/mm_cache/ C.INSTANCE_ID:/workspace/agentic_trading/experiments/mm_cache/
```

SSH into the machine and run:

```bash
cd /workspace/agentic_trading
source .venv/bin/activate
python -m experiments.run_mm_deep_experiment \
  --config experiments/configs/icaif_real_news_costaware_main.json \
  --device auto
```

Copy results back:

```bash
.venv/bin/vastai copy C.INSTANCE_ID:/workspace/agentic_trading/experiments/results_mm/ local:experiments/results_mm_vast/
```

Destroy the instance when finished:

```bash
.venv/bin/vastai destroy instance INSTANCE_ID
```

Stopping an instance pauses compute billing but disk charges can continue. Destroying is the clean default for one-off runs.

Do not copy files into `/root` or `/`; Vast copy uses SSH authentication, and changing root directory permissions can break later copy and login operations.

## Hugging Face Artifact Workflow

Use Hugging Face as a public artifact release target, not as a dump for full raw caches.

First log in:

```bash
source .venv/bin/activate
hf auth login
```

Upload curated public artifacts:

```bash
HF_REPO_ID=xiaoxiaoshikui/agentic-trading-results \
HF_REPO_TYPE=dataset \
bash deploy/upload_hf_artifacts.sh
```

For a private staging repo:

```bash
HF_REPO_ID=xiaoxiaoshikui/agentic-trading-results \
HF_REPO_TYPE=dataset \
HF_PRIVATE=1 \
bash deploy/upload_hf_artifacts.sh
```

The upload script stages:

- `README.md`
- `LICENSE`
- `docs/figures/cgcma_architecture.png`
- public CSV/JSON/Markdown result files under `paper/`
- selected LaTeX source files
- ICAIF experiment configs

It does not stage raw data, `.env`, large caches, private reviews, rebuttals, or submission packages.

## Recommended Next Steps

1. Install or restore Vast.ai CLI and authenticate it.
2. Log into Hugging Face with `.venv/bin/hf auth login`.
3. Run the four-way image smoke test above, using `deploy/vast_ssh_repair.sh` for the repair-on variants.
4. If the new image only works with repair, fix the Dockerfile so `/root` is not group/world-writable and `/root/.ssh` is not created with unsafe modes.
5. Launch a low-cost Vast smoke instance and verify `deploy/vast_bootstrap.sh`.
6. Copy only the data needed for `icaif_real_news_costaware_main`.
7. Run one full GPU experiment and download the result folder.
8. Regenerate paper tables from the Vast result folder.
9. Upload curated public artifacts to Hugging Face Hub.
10. Add the Hugging Face artifact URL to `README.md`.
