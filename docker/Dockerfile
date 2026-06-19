# Linux amd64 PyInstaller build environment (matches GitHub Actions ubuntu-22.04 / Python 3.10).
FROM python:3.10-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    binutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Pre-install runtime deps; build_submission.sh refreshes if requirements.txt changes.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt pyinstaller

CMD ["/work/docker/build_submission.sh"]
