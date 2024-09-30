# Smooth k-nearest neighbors classifier

## Comparison
![Classifier comparison](docs/plot_classifier_comparison.png)

## Results
| | Make Moons | Make Circles | Linearly Separable | Iris | Wine | Average | 
| - | - | - | - | - | - | - | 
| Smooth Nearest Neighbors | 0.95 | 0.925 | 0.95 | 0.9833333333333333 | 0.9722222222222222 | 0.9561111111111111 | 
| Nearest Neighbors | 0.975 | 0.925 | 0.95 | 0.9833333333333333 | 0.9444444444444444 | 0.9555555555555555 | 
| Gaussian Process | 0.975 | 0.9 | 0.925 | 0.9833333333333333 | 0.9583333333333334 | 0.9483333333333333 | 
| Neural Net | 0.9 | 0.875 | 0.95 | 0.9833333333333333 | 0.9861111111111112 | 0.9388888888888889 | 
| AdaBoost | 0.925 | 0.85 | 0.95 | 0.9833333333333333 | 0.9722222222222222 | 0.9361111111111111 | 
| Decision Tree | 0.95 | 0.775 | 0.95 | 0.9833333333333333 | 0.9444444444444444 | 0.9205555555555556 | 
| Random Forest | 0.95 | 0.75 | 0.95 | 0.9833333333333333 | 0.9583333333333334 | 0.9183333333333333 | 
| Naive Bayes | 0.875 | 0.7 | 0.95 | 0.9666666666666667 | 1.0 | 0.8983333333333334 | 
| QDA | 0.85 | 0.725 | 0.925 | 0.9833333333333333 | 0.9722222222222222 | 0.8911111111111112 | 
| RBF SVM | 0.975 | 0.875 | 0.95 | 0.9833333333333333 | 0.375 | 0.8316666666666667 | 
| Linear SVM | 0.875 | 0.4 | 0.925 | 0.9 | 0.9861111111111112 | 0.8172222222222223 | 

## Installation
```
pip install smooth-knn
```

## Usage
```python
from smooth_knn.classifier import SmoothKNeighborsClassifier

clf = SmoothKNeighborsClassifier()
clf.fit(X, y)
```
