# XRD ETL Pipeline

Reproducible ETL pipeline for experimental powder XRD data from the
`hodge_alwen_1` subset of the opXRD dataset.

## Running the pipeline

## Data source and current limitation

This project uses the `hodge_alwen_1` subset of the opXRD experimental
powder XRD dataset. The subset contains 169 JSON files and is approximately
10 MB.

The preferred design for this project is to retrieve the raw data
programmatically from an external source rather than store raw data in the
Git repository. However, opXRD is currently distributed as a much larger
archive, while this ETL uses only a small subset of the full dataset.

I was able to recover my huggingface account so the data was moved there.
