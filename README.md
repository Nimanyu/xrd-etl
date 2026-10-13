# XRD ETL Pipeline

Reproducible ETL pipeline for experimental powder X-ray diffraction (XRD)
data from the `hodge_alwen_1` subset of the opXRD dataset.

The pipeline downloads the raw data, verifies its checksum, extracts and
transforms the JSON records, and loads the processed data into a normalized
DuckDB database.

## Running the pipeline

From the repository root, run:

```bash
just
