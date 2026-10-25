The pipeline also converts the DuckDB database into model-ready SafeTensor files.

Run:
just

This will produce:
data/dataset.db
data/inputs/train.safetensors
data/inputs/eval.safetensors

Each SafeTensor file contains:
	two_theta: padded 2θ values 
	intensity: padded XRD intensities
	mask: identifies real data (True) and padding (False)
	group_id: identifies the original XRD pattern
	cv_fold: cross-validation fold assignment

The dataset contains 169 XRD patterns and is split deterministically into:
	Train: 135 patterns
	Eval:   34 patterns

Splitting is performed at the pattern_id level so that all points from one XRD pattern remain in the same dataset and cannot leak between training and evaluation sets. Training patterns are also assigned deterministically to five cross-validation folds.

Patterns have different numbers of measured points, so tensors are padded to the maximum length of 3050 points. A mask distinguishes padding from measured data.

To validate the generated files, run:
just check

The check verifies that the SafeTensor files load correctly, tensor dimensions agree, train and evaluation groups do not overlap, cross-validation assignments are valid, and all database patterns are accounted for.
Repeated generation produces identical SafeTensor files.
