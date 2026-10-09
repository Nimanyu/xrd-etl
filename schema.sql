CREATE TABLE patterns (
    pattern_id VARCHAR PRIMARY KEY,
    source_collection VARCHAR,
    source_file VARCHAR,

    two_theta DOUBLE[],
    intensity DOUBLE[],

    is_simulated BOOLEAN,
    primary_wavelength DOUBLE,
    secondary_wavelength DOUBLE,
    crystallite_size DOUBLE,
    temperature_c DOUBLE,

    institution VARCHAR,
    contributor_name VARCHAR,
    original_file_format VARCHAR,
    measurement_date VARCHAR,
    xrdpattern_version VARCHAR,

    raw_label VARCHAR,
    raw_metadata VARCHAR
);

CREATE TABLE phases (
    pattern_id VARCHAR,
    phase_index INTEGER,

    chemical_composition VARCHAR,
    phase_fraction DOUBLE,
    spacegroup INTEGER,

    a DOUBLE,
    b DOUBLE,
    c DOUBLE,

    alpha DOUBLE,
    beta DOUBLE,
    gamma DOUBLE,

    crystal_system VARCHAR,

    PRIMARY KEY (pattern_id, phase_index),
    FOREIGN KEY (pattern_id) REFERENCES patterns(pattern_id)
);
