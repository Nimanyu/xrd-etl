import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import duckdb

DATA_DIR = Path("data/raw")
RAW_DIR = DATA_DIR / "hodge_alwen_1"
ARCHIVE = DATA_DIR / "hodge_alwen_1.zip"

DATA_URL = (
    "https://huggingface.co/datasets/nimanyuj/"
    "xrd-etl-data/resolve/main/hodge_alwen_1.zip"
)

EXPECTED_SHA256 = (
    "39fd40e1bd6103e6a0c7f2c1ec3f8c1895fb423277e46a1258922660d5f3dd98"
)

MAX_PATTERNS = 169

def parse_json_string(value):
    """Convert JSON stored as a string into a Python dictionary."""
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    return json.loads(value)

def sha256sum(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def ensure_raw_data():
    if RAW_DIR.exists():
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not ARCHIVE.exists():
        print("[etl] Downloading XRD source data", flush=True)
        urllib.request.urlretrieve(DATA_URL, ARCHIVE)

    actual_hash = sha256sum(ARCHIVE)

    if actual_hash != EXPECTED_SHA256:
        ARCHIVE.unlink(missing_ok=True)
        raise RuntimeError(
            f"Dataset checksum mismatch: {actual_hash}"
        )

    print("[etl] Dataset checksum verified", flush=True)
    print("[etl] Extracting source data", flush=True)

    with zipfile.ZipFile(ARCHIVE) as zf:
        zf.extractall(DATA_DIR)

    # The uploaded archive may contain the original directory hierarchy.
    if not RAW_DIR.exists():
        matches = [
            path
            for path in DATA_DIR.rglob("hodge_alwen_1")
            if path.is_dir()
        ]

        if len(matches) != 1:
            raise RuntimeError(
                "Could not uniquely locate hodge_alwen_1 after extraction"
            )

        matches[0].rename(RAW_DIR)

def main(out):
    ensure_raw_data()
    files = sorted(RAW_DIR.rglob("*.json"))

    if not files:
        raise RuntimeError(f"No JSON files found in {RAW_DIR}")

    print(f"[etl] Searching {len(files)} JSON files for {MAX_PATTERNS} valid patterns", flush=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # Delete the old database so every run starts cleanly.
    Path(out).unlink(missing_ok=True)

    db = duckdb.connect(out)

    # Create tables using schema.sql.
    db.execute(Path("schema.sql").read_text())

    pattern_rows = []
    point_rows = []
    phase_rows = []

    for path in files:
        with open(path) as f:
            raw = json.load(f)

        two_theta = raw["two_theta_values"]
        intensity = raw["intensities"]

        # X and Y must have the same number of points.
        if len(two_theta) != len(intensity):
            continue

        label = parse_json_string(raw.get("label"))
        metadata = parse_json_string(raw.get("metadata"))
        xray = parse_json_string(label.get("xray_info"))

        # Example:
        # breitung_schweidler_1/pattern_19
        pattern_id = f"{path.parent.name}/{path.stem}"

        pattern_rows.append(
            (
                pattern_id,
                path.parent.name,
                str(path),
                label.get("is_simulated"),
                xray.get("primary_wavelength"),
                xray.get("secondary_wavelength"),
                label.get("crystallite_size"),
                label.get("temp_in_celcius"),
                metadata.get("institution"),
                metadata.get("contributor_name"),
                metadata.get("original_file_format"),
                metadata.get("measurement_date"),
                metadata.get("xrdpattern_version"),
            )
        )

        for point_index, (theta, counts) in enumerate(
            zip(two_theta, intensity)
        ):
            point_rows.append(
                (
                    pattern_id,
                    point_index,
                    theta,
                    counts,
                )
        )

        # One XRD pattern can contain multiple phases.
        for phase_index, phase_string in enumerate(label.get("phases", [])):
            phase = parse_json_string(phase_string)

            lengths = parse_json_string(phase.get("lengths"))
            angles = parse_json_string(phase.get("angles"))

            phase_rows.append(
                (
                    pattern_id,
                    phase_index,
                    phase.get("chemical_composition"),
                    phase.get("phase_fraction"),
                    phase.get("spacegroup"),
                    lengths.get("a"),
                    lengths.get("b"),
                    lengths.get("c"),
                    angles.get("alpha"),
                    angles.get("beta"),
                    angles.get("gamma"),
                )
            )
        if len(pattern_rows) % 10 == 0:
            print(f"[etl] Processing pattern {len(pattern_rows)}", flush=True)

        if len(pattern_rows) >= MAX_PATTERNS:
            break
    if not pattern_rows:
        raise RuntimeError("No valid XRD patterns were found")

    db.execute("BEGIN TRANSACTION")

    db.executemany(
        """
        INSERT INTO patterns VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        pattern_rows,
    )

    db.executemany(
        """
        INSERT INTO pattern_points VALUES (
            ?, ?, ?, ?
        )
        """,
        point_rows,
    )

    if phase_rows:
        db.executemany(
            """
            INSERT INTO phases VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            phase_rows,
        )

    db.execute("COMMIT")

    print(f"[etl] Loaded {len(pattern_rows)} patterns")
    print(f"[etl] Loaded {len(point_rows)} pattern points")
    print(f"[etl] Loaded {len(phase_rows)} phases")
    print(f"[etl] Database written to {out}")

    db.close()


if __name__ == "__main__":
    main(sys.argv[1])
