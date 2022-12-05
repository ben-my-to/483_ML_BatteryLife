from .decision_tree import *


from statistics import mean
from sklearn.metrics import mean_squared_error


np.random.seed(42)


class RandomForest(DecisionTreeClassifier, DecisionTreeRegressor):
    def __init__(self, n_estimators=5, n_sample=0, criterion={}):
        self.n_estimators = n_estimators
        self.n_sample = n_sample
        self.forest = [
            DecisionTreeRegressor(criterion=criterion) for _ in range(n_estimators)
        ]

    def __repr__(self):
        return "\n".join(repr(dt) for dt in self.forest)

    def sub_sample(self, X, n_sample=2):
        """
        Enforces feature randomness
        """
        return np.random.choice(X.columns.to_numpy(), n_sample, replace=False)

    def bootstrap_sample(self, X, y, n_sample, key=True):
        feature_subset = self.sub_sample(X, int(np.log2(len(X.columns))))
        d = pd.concat([X, y], axis=1)
        d = d.sample(n=n_sample, replace=key, random_state=42)
        return d.iloc[:, :-1][feature_subset], d.iloc[:, -1]

    def fit(self, X, y):
        for tree in self.forest:
            tree.fit(*self.bootstrap_sample(X, y, self.n_sample))
        return self

    def predict(self, x):
        assert all(isinstance(model, DecisionTreeRegressor) for model in self.forest)
        return mean([dt.predict(x).feature for dt in self.forest])

    def predict_all(self, X):
        assert all(isinstance(model, DecisionTreeRegressor) for model in self.forest)
        return [self.predict(X.iloc[x].to_frame().T) for x in range(len(X))]

    def score(self, X, y):
        y_hat = [self.predict(X.iloc[x].to_frame().T) for x in range(len(X))]
        return mean_squared_error(y, y_hat, squared=False)
