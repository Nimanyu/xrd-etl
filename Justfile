default: pipeline

pipeline:
    podman run --rm -v .:/work -w /work $(podman build -q .) make all

check: pipeline
    podman run --rm -v .:/work -w /work $(podman build -q .) \
        uv run python -m mseml.check \
        data/dataset.db \
        data/inputs/train.safetensors \
        data/inputs/eval.safetensors

hpc *ARGS:
    uv run python hpc/submit.py {{ARGS}}
