

# Makefile for XRD ETL, model training, analysis, and report

CONFIG ?= config.yaml

all: report/main.pdf

# Build DuckDB database
data/dataset.db: src/mseml/etl.py schema.sql
	uv run python -m mseml.etl $@

# Convert database into train/eval SafeTensors
data/inputs/train.safetensors data/inputs/eval.safetensors &: data/dataset.db src/mseml/tensors.py
	uv run python -m mseml.tensors data/dataset.db data/inputs

# Train CNN and evaluate CNN, dummy baseline, and positive control
results/results.json: data/inputs/train.safetensors data/inputs/eval.safetensors src/mseml/model.py
	uv run python -m mseml.model

# Generate training curve
results/training_curve.png: results/results.json src/mseml/plot.py
	uv run python -m mseml.plot

# Compile report
report/main.pdf: results/results.json results/training_curve.png report/main.typ
	typst compile --root . report/main.typ $@
