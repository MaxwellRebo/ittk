ittk: Information Theory Toolkit
====
[![Code Climate](https://codeclimate.com/github/MaxwellRebo/ittk/badges/gpa.svg)](https://codeclimate.com/github/MaxwellRebo/ittk)
[![Join the chat at https://gitter.im/MaxwellRebo/ittk](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/MaxwellRebo/ittk?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

  Information-theoretic methods in Python.  Intended for use in data analysis and machine learning applications.

  Kept lean and very simple for transparency and ease of integration in other projects.  Just import into your project and call the appropriate methods wherever you need to - no need to delve into the esoteric math books.

  Please ping me if you find any errors.  These functions have been tested against both Matlab and R implementation of the same kind, and found to be generally sound as of this writing.

  If you have a suggestion for an algorithm or metric you'd like to see added here, please let me know and I'm happy to add it.

###Testing
    
To run tests, simply do:
```
nosetests
```

This will automatically run all of the tests in the tests/ directory.
  
###Usage examples

All of these examples assume discrete variables.

Make sure you're using numpy arrays.

Get the probability of each variable occuring:

```python
import ittk

ittk.probs([1,2,3])
array([ 0.,  0.33333333,  0.33333333,  0.33333333])
```

Get the entropy of a variable from some discrete observations:

```python
X = numpy.array([7, 7, 7])
ittk.entropy(X)
0.0

Y = numpy.array([0,1,2,3])
ittk.entropy(Y)
2.0
```

Get the mutual information and variation of information between two variables:

```python
X = numpy.array([7,7,7,3])
Y = numpy.array([0,1,2,3])
ittk.mutual_information(X, Y)
0.8112781244591329

ittk.information_variation(X, Y)
1.1887218755408671

A = numpy.array([1,2,3,4])
B = numpy.array([1,2,3,4])
ittk.mutual_information(A, B)
2.0
```

Note that the above is not normalized. To normalize it to [0, 1], pass normalized as True:

```python
ittk.mutual_information(A, B, normalized=True)
1.0
```
  
###Dependencies

  -numpy
  -nose
  
  Do:
  
  ```
  pip install -r requirements.txt
  ```
  
#####License: MIT
  
