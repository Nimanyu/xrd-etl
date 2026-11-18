// EMA 5937 — Model & Report Part I
// Values are read automatically from results/results.json.

#let R = json("/results/results.json")
#let f(x, d: 3) = str(calc.round(x, digits: d))

#set document(
  title: "Predicting Al Phase Fraction from XRD Patterns with a 1D CNN",
  author: "Nimanyu Joshi",
)

#set page(margin: 2.5cm, numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(17pt, weight: "bold")[
    Predicting Al Phase Fraction from XRD Patterns with a 1D CNN
  ]
  #v(0.4em)
  Nimanyu Joshi — EMA 5937 Special Topics: ML for Materials Science
]

= Motivation

X-ray diffraction (XRD) patterns contain information about the crystalline
phases present in a material. Conventional quantitative phase analysis often
requires explicit peak fitting or refinement. This project investigates whether
a machine-learning model can instead learn a direct relationship between a
full XRD intensity pattern and a quantitative phase fraction.

For this first model milestone, the task is regression: the input is the full
XRD intensity pattern and the target is the Al phase fraction associated with
that pattern.

= Data and preprocessing

The current dataset contains 169 XRD patterns from the opXRD dataset. Each
pattern contains 3050 intensity values. The patterns are divided
deterministically into 135 training samples and 34 held-out evaluation samples.

The data are stored in a normalized DuckDB database and converted into
SafeTensors for model training. Before entering the model, each intensity
pattern is normalized by its maximum intensity.

= Model

A one-dimensional convolutional neural network (1D CNN) is used for regression.
The network contains two one-dimensional convolutional layers with ReLU
activations, followed by pooling and a linear output layer. The model takes the
complete intensity sequence as input and predicts one scalar Al phase fraction.

The network is trained using the Adam optimizer and mean squared error (MSE)
loss. Performance on the held-out evaluation set is reported using mean
absolute error (MAE).

Two controls are included. The negative control is a dummy predictor that
always predicts the mean Al phase fraction of the training set. The positive
control is an oracle that predicts the held-out ground-truth target exactly,
and therefore has zero MAE. The oracle is not a predictive model; it is included
only as a sanity check for the evaluation metric.

= Results

#figure(
  image("/results/training_curve.png", width: 85%),
  caption: [Training MSE of the 1D CNN as a function of epoch.]
) <fig-training>

#figure(
  table(
    columns: 2,
    align: (left, right),
    stroke: none,
    table.hline(),
    table.header([*Model / control*], [*Held-out MAE*]),
    table.hline(stroke: 0.5pt),
    [Positive control (oracle)], [#f(R.positive_control_test_mae)],
    [Dummy predictor], [#f(R.dummy_test_mae)],
    [1D CNN], [#f(R.cnn_test_mae)],
    table.hline(),
  ),
  caption: [Held-out mean absolute error for the CNN and controls.]
) <tab-results>

The current CNN obtains a held-out MAE of #f(R.cnn_test_mae), compared with
#f(R.dummy_test_mae) for the dummy predictor. Therefore, the current CNN does
not yet outperform the simple training-mean baseline.

The purpose of this milestone is to establish the complete model-training and
evaluation pipeline rather than to optimize predictive performance. The
training curve in @fig-training confirms that the model is being optimized,
while @tab-results shows that further model or preprocessing improvements are
required to obtain useful generalization.

= Next steps

Future work will investigate improved preprocessing and model design. In
particular, XRD patterns from different experimental sources may use different
two-theta grids. Resampling patterns onto a common two-theta grid would allow
the model to interpret convolution positions consistently across a larger
dataset.

Additional prediction tasks, including crystallographic classification, can
also be explored as the dataset is expanded.

= Reproducibility

The project is designed as a dependency-driven pipeline. The `just` command
runs the containerized Makefile workflow, beginning with the source data and
DuckDB database, generating training and evaluation tensors, training and
evaluating the model, generating analysis results and figures, and finally
compiling this report.
