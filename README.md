# Machine Learning for Physics
 
Python modules to perform analysis of high energy physics data.

## [custom_dataset.py](python/custom_dataset.py)
- Create a dataset of training and validation examples with simulations of the CMS electromagnetic calorimeter.

## [custom_models.py](python/custom_models.py)
- Create models in [TensorFlow](https://www.tensorflow.org).

## [custom_estimator.py](python/custom_estimator.py)
- Build a classifier using [tf.estimator](https://www.tensorflow.org/api_docs/python/tf/estimator) API.

# Electron, Photon and Pion Classification with Neural Networks 
Each classification model was trained in batches of 128 images during 10K steps (64 epochs).
The energy threshold vary from 0 to 30 GeV in steps of 10 GeV.
In terms of classification accuracy, the convolutional neural network outperforms the shallow model.
The following table sumarizes the results.

<table>
  <tr>
    <th colspan="6"><span style="font-weight:bold">Accuracy Results</span></th>
  </tr>
  <tr>
    <td>Model \ Energy threshold</td>
    <td>0</td>
    <td>10</td>
    <td>20</td>
    <td>30</td>
  </tr>
  <tr>
    <td><a href="https://github.com/jruizvar/ml-physics/blob/master/python/custom_models.py#L6-L21">Shallow NN</a></td>
    <td>0.895</td>
    <td>0.903</td>
    <td>0.900</td>
    <td>0.904</td>
  </tr>
  <tr>
    <td><a href="https://github.com/jruizvar/ml-physics/blob/master/python/custom_models.py#L24-L58">Convolutional NN</a></td>
    <td>0.914</td>
    <td>0.933</td>
    <td>0.926</td>
    <td>0.922</td>
  </tr>
</table>
