import json
import sys
from pathlib import Path

import duckdb

RAW_DIR = Path("source_data/opxrd/hodge_alwen_1")
MAX_PATTERNS = 169


def parse_json_string(value):
    """Convert JSON stored as a string into a Python dictionary."""
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    return json.loads(value)


def main(out):
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
                two_theta,
                intensity,
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
                raw.get("label"),
                raw.get("metadata"),
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
                    phase.get("crystal_system"),
                )
            )
        if len(pattern_rows) % 10 == 0:
            print(f"[etl] Processing pattern {len(pattern_rows)}", flush=True)

        if len(pattern_rows) >= MAX_PATTERNS:
            break
    if not pattern_rows:
        raise RuntimeError("No valid XRD patterns were found")
    db.executemany(
        """
        INSERT INTO patterns VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        pattern_rows,
    )
    if phase_rows:
        db.executemany(
            """
            INSERT INTO phases VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            phase_rows,
        )

    print(f"[etl] Loaded {len(pattern_rows)} patterns")
    print(f"[etl] Loaded {len(phase_rows)} phases")
    print(f"[etl] Database written to {out}")

    db.close()


if __name__ == "__main__":
    main(sys.argv[1])
