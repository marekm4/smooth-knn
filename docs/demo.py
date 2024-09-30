X = [[1], [1], [3], [3], [4], [4], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7]]
y = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

from smooth_knn.classifier import SmoothKNeighborsClassifier

clf = SmoothKNeighborsClassifier()
clf.fit(X, y)

print(clf.score(X, y))
