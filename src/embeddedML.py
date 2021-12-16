import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.smpickle import load_pickle

class LinearRegression():
    def __init__(self, model_path, scaler=None, inv_scaler=None):
        self.model = load_pickle(model_path)
        self.scaler = scaler
        self.inv_scaler = inv_scaler

    def pipeline(self, X, add_const=False):
        if self.scaler:
            X = self.scaler(X)
        if add_const:
            X = sm.add_constant(X, has_constant='add')
        return X

    def predict(self, X, add_const=True):
        X = self.pipeline(X, add_const)
        Y = self.model.predict(X)
        if self.inv_scaler:
            Y = self.inv_scaler(Y)
        return Y

    def summary(self):
        return self.model.summary()