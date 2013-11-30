ittk: Information Theory Toolkit
====

  Information-theoretic methods in Python.  Intended for use in data analysis and machine learning applications.

  Kept lean and very simple for transparency and ease of integration in other projects.  Just import into your project and call the appropriate methods wherever you need to - no need to delve into the esoteric math books.

  Please ping me if you find any errors.  These functions have been tested against both Matlab and R implementation of the same kind, and found to be generally sound as of this writing.

  TODO: Expand to more complex data types; helper methods to format data; parallel/many-core versions; additional useful information-theoretic and probabilistic methods.

  If you have a suggestion for an algorithm you'd like to see added here, please let me know and I'm happy to add it!
  
###Usage examples

All of these examples assume discrete variables.

First, do `import ittk`.

Just get the probability of each variable occuring:

```python
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

Note that the above is not normalized.
  
###Dependencies

  -numpy
  
#####License: MIT
  
