import hyperopt
from hyperopt import fmin, tpe, hp, Trials

classifiers = {
    'lr': LogisticRegression(),
    'knn': KNeighborsClassifier(3),
    'svc_lin': SVC(kernel="linear", C=0.025),
    'svc': SVC(gamma=2, C=1),
    'rf': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    'mlp': MLPClassifier(alpha=1),
    'ada': AdaBoostClassifier(),
    'nb': GaussianNB()
}