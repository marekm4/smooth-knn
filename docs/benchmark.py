import numpy as np
from sklearn.datasets import make_circles, make_classification, make_moons, load_iris, load_wine
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.smooth_knn.classifier import SmoothKNeighborsClassifier

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "Smooth Nearest Neighbors",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    SmoothKNeighborsClassifier(),
]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
iris = load_iris()
wine = load_wine()

datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
    (iris.data, iris.target),
    (wine.data, wine.target),
]

dataset_names = [
    "Make Moons",
    "Make Circles",
    "Linearly Separable",
    "Iris",
    "Wine",
]

scores = {}
for ds_cnt, ds in enumerate(datasets):
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    for name, clf in zip(names, classifiers):
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if name not in scores:
            scores[name] = []
        scores[name].append(score)

# table
print(end="| | ")
for name in dataset_names + ["Average"]:
    print(name, end=" | ")
print()

print(end="| - | ")
for name in dataset_names + ["Average"]:
    print("-", end=" | ")
print()

for name, score in sorted(scores.items(), key=lambda x: sum(x[1]), reverse=True):
    print(end="| ")
    print(name, end=" | ")
    for s in score:
        print(round(s, 4), end=" | ")
    print(round(np.sum(score) / len(datasets), 4), end=" | ")
    print()
