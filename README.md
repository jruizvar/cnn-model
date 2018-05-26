# Machine Learning for Physics
 
Python modules to perform analysis of high energy physics data.

## [custom_dataset.py](python/custom_dataset.py)
- Create a dataset of training and validation examples with simulations of the CMS electromagnetic calorimeter.

## [custom_models.py](python/custom_models.py)
- Create models in [TensorFlow](https://www.tensorflow.org).

## [custom_estimator.py](python/custom_estimator.py)
- Build a classifier using [tf.estimator](https://www.tensorflow.org/api_docs/python/tf/estimator) API.

# Results

The following table sumarizes the results of the classification models.

<table>
  <tr>
    <th colspan="6"><span style="font-weight:bold">Accuracy results</span></th>
  </tr>
  <tr>
    <td><span style="font-weight:bold">Model \ Energy threshold</span></td>
    <td>0<br></td>
    <td>10<br></td>
    <td>20<br></td>
    <td>30<br></td>
    <td>40</td>
  </tr>
  <tr>
    <td><a href="https://github.com/jruizvar/ml-physics/blob/master/python/custom_models.py#L6-L21">Shallow NN</a></td>
    <td>0.70<br></td>
    <td>0.75<br></td>
    <td>0.80<br></td>
    <td>0.85<br></td>
    <td>0.90</td>
  </tr>
  <tr>
    <td><a href="https://github.com/jruizvar/ml-physics/blob/master/python/custom_models.py#L24-L58">Convolutional NN</a></td>
    <td>0.909</td>
    <td>0.926<br></td>
    <td>0.911<br></td>
    <td>0.918</td>
    <td>0.922</td>
  </tr>
</table>
