# Energy Regression

The goal is to train a regression model that minimizes the experimental errors associated
with the energy reconstruction in the CMS electromagnetic calorimeter.

The results shown below correspond to the sample `eplus_Ele-Eta0PhiPiOver2-Energy20to100_V2.npy`.

![](notebooks/gen_energy.png)

## Baseline

The root mean squared error [RMSE](python/custom_estimator.py#L58) measures the discrepancy
between the reconstructed energy and the truth value. An initial baseline is the following:

- RMSE = 0.171

![](notebooks/reco_vs_gen_energy.png)

## Model Evaluation
Our results demonstrate the good performance of a convolutional model at minimizing the [RMSE](python/custom_estimator.py#L58).

<table>
  <tr>
    <th colspan="6"><span style="font-weight:bold">RMSE results</span></th>
  </tr>
  <tr>
    <td>Model \ Energy threshold</td>
    <td>0 GeV</td>
    <td>10 GeV</td>
    <td>20 GeV</td>
    <td>30 GeV</td>
  </tr>
  <tr>
    <td><a href="https://github.com/jruizvar/ml-physics/blob/master/python/custom_models.py#L6-L21">Shallow NN</a></td>
    <td>0.194</td>
    <td>0.177</td>
    <td>0.485</td>
    <td>0.351</td>
  </tr>
  <tr>
    <td><a href="https://github.com/jruizvar/ml-physics/blob/master/python/custom_models.py#L24-L59">Convolutional NN</a></td>
    <td>0.122</td>
    <td>0.068</td>
    <td>0.077</td>
    <td>0.075</td>
  </tr>
</table>

## Learning Curve
The [loss](python/custom_estimator.py#L45) is defined by the mean squared error between the labels and the model predictions.
The horizonal axis represents the number of [steps](python/custom_estimator.py#L80). One epoch is equivalent to 100 steps.

The blue (orange) curve corresponds to the validation (training) sample. After one epoch the model seems to converge.

![](doc/learning_curve.png)

The evolution of the [RMSE](python/custom_estimator.py#L58) calculated for the validation sample is shown below.

![](doc/rmse.png)

## TensorBoard Graph

The complete graph associated to the convolutional model is displayed below.

![](doc/graph.png)

## Python Modules

## [custom_dataset.py](python/custom_dataset.py)
- Create a dataset of training and validation examples with simulations of the CMS electromagnetic calorimeter.

## [custom_models.py](python/custom_models.py)
- Create models in [TensorFlow](https://www.tensorflow.org).

## [custom_estimator.py](python/custom_estimator.py)
- Build a regression model using [tf.estimator](https://www.tensorflow.org/api_docs/python/tf/estimator) API.
