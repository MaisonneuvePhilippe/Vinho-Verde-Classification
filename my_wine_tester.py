from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import PolynomialFeatures
from wine_testers import WineTester
from sklearn.decomposition import PCA
import pandas as pd


COL_NAMES = [
    "id",
    "color",
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


class MyWineTester(WineTester):
    def __init__(self):

        self.poly = PolynomialFeatures(degree=2)
        self.pca = PCA(n_components=28)
        self.regressor = ExtraTreesRegressor(
            n_estimators=2250,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            max_depth=160,
            bootstrap=False,
        )

    def train(self, X_train, y_train):

        """
        train the current model on train_data
        :param X_train: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :param y_train: 2D array of labels.
                each line is a different example.
                the first column is the example ID.
                the second column is the example label.
        """
        X_train = pd.DataFrame(data=X_train, columns=COL_NAMES)
        y_train = pd.DataFrame(data=y_train, columns=["id", "quality"])
        y_train = y_train.drop(columns="id")
        X_train["is_white"] = (X_train["color"] == "white").astype(int)
        X_train = X_train.drop(columns=["color", "id"])

        X_train = self.poly.fit_transform(X_train)
        X_train = self.pca.fit_transform(X_train)
        self.regressor.fit(X_train, y_train)

    def predict(self, X_data):
        """
        predict the labels of the test_data with the current model
        and return a list of predictions of this form:
        [
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            ...
        ]
        :param X_data: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :return: a 2D list of predictions with 2 columns: ID and prediction
        """
        X_data = pd.DataFrame(data=X_data, columns=COL_NAMES)
        X_data["is_white"] = (X_data["color"] == "white").astype(int)
        X_data = X_data.drop(columns=["color", "id"])
        X_data = self.poly.transform(X_data)
        X_data = self.pca.transform(X_data)

        prediction = self.regressor.predict(X_data)
        prediction = list(
            zip(range(len(prediction)), [int(round(num, 0)) for num in prediction])
        )

        return prediction
