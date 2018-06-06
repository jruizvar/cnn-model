# Energy Regression

## Graph

![](doc/graph.png)

## Learning Curve
The blue (orange) curve corresponds to the validation (training) sample.

![](doc/learning_curve.png)

## Root Mean Squared Error

![](doc/rmse.png)

<table>
  <tr>
    <th colspan="6"><span style="font-weight:bold">RMSE Results</span></th>
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
    <td>0.122</td>
    <td>0.067</td>
    <td>0.060</td>
    <td>0.070</td>
  </tr>
  <tr>
    <td><a href="https://github.com/jruizvar/ml-physics/blob/master/python/custom_models.py#L24-L59">Convolutional NN</a></td>
    <td>0.122</td>
    <td>0.067</td>
    <td>0.060</td>
    <td>0.070</td>
  </tr>
</table>

## Python Modules

## [custom_dataset.py](python/custom_dataset.py)
- Create a dataset of training and validation examples with simulations of the CMS electromagnetic calorimeter.

## [custom_models.py](python/custom_models.py)
- Create models in [TensorFlow](https://www.tensorflow.org).

## [custom_estimator.py](python/custom_estimator.py)
- Build a regressor using [tf.estimator](https://www.tensorflow.org/api_docs/python/tf/estimator) API.
