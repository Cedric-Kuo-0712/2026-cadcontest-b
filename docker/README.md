# Docker build — Linux executable

Build a **Linux amd64** contest binary from macOS (or any host with Docker).

## Prerequisites

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
2. This repo checked out at the project root (with `src/`, `requirements.txt`, `eval.py`)

## Quick start (from repo root)

```bash
chmod +x docker/build_linux.sh build_linux.sh
./build_linux.sh
```

Output lands in `submission/`:

```
submission/
├── regr_fail_bucketing   # Linux ELF binary (PyInstaller one-file)
├── README.md
├── requirements.txt      # fallback if binary fails on judge machine
└── src/                  # fallback Python entry point
```

## Step-by-step

### 1. Build the Docker image

Uses `python:3.10-bookworm` + PyInstaller + your `requirements.txt`.

```bash
docker build --platform linux/amd64 \
  -t regr-fail-bucketing-linux-build \
  -f docker/Dockerfile .
```

### 2. Run the build inside the container

Mount the repo so PyInstaller sees the latest source:

```bash
docker run --rm --platform linux/amd64 \
  -v "$(pwd):/work" \
  -w /work \
  regr-fail-bucketing-linux-build \
  /work/docker/build_submission.sh
```

### 3. Verify the binary

```bash
file submission/regr_fail_bucketing
# expect: ELF 64-bit LSB executable, x86-64

submission/regr_fail_bucketing \
  --input B_samples_20260516/problem/benchmark_set_1/input.csv \
  --output /tmp/out.csv --k 2

python3 eval.py --output /tmp/out.csv \
  --golden B_samples_20260516/problem/benchmark_set_1/golden.csv
```

### 4. Package for upload

```bash
zip -r submission-linux.zip submission
```

## Files in this folder

| File | Purpose |
|------|---------|
| `Dockerfile` | Linux build environment |
| `regr_fail_bucketing.spec` | PyInstaller configuration |
| `build_submission.sh` | Runs inside container: pip + pyinstaller + assemble `submission/` |
| `build_linux.sh` | Host wrapper: `docker build` + `docker run` |

## Notes

- PyInstaller binaries are **OS-specific**. Build with this flow before submitting to a Linux judge machine.
- First Docker build may take several minutes (downloads Python image + pip packages).
- Re-runs are faster: only `docker run` is needed after the image exists.
- On Apple Silicon, `--platform linux/amd64` ensures an x86_64 binary (contest default).

## Alternative: GitHub Actions

Push to `main` / `master` / `bsc/v1` and download the `submission-linux` artifact from Actions (`.github/workflows/build-linux.yml`).
