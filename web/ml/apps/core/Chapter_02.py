from .utlities import Algo
import os
import matplotlib.pyplot as plt
import hashlib
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class Algo2(Algo):
    def __init__(self):
        super().__init__(chapter_id="CH02_end_to_end_project", to_data_path="housing", target_field="median_house_value")
        csv_path = os.path.join(self.TO_DATA_PATH, "housing.csv")
        if not os.path.isfile(csv_path):
            self.DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
            self.HOUSING_URL = self.DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
            self.fetch_tgz_data(self.HOUSING_URL, "housing")
        self.load_csv_data("housing")


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True, algo=None): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.ALGO = algo
        if algo:
            self.ROOMS_IX, self.BEDROOMS_IX, self.POPULATION_IX, self.HOUSEHOLD_IX = [
                list(self.ALGO.TRAIN_DATA.columns).index(col)
                for col in ("total_rooms", "total_bedrooms", "population", "households")]
        else:
            self.ROOMS_IX, self.BEDROOMS_IX, self.POPULATION_IX, self.HOUSEHOLD_IX = [3, 4, 5, 6]

        # print('CombinedAttributesAdder: the indexes of the fields:')
        # print(self.ROOMS_IX, self.BEDROOMS_IX, self.POPULATION_IX, self.HOUSEHOLD_IX)
        # print('-'*100)

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, self.ROOMS_IX] / X[:, self.HOUSEHOLD_IX]
        population_per_household = X[:, self.POPULATION_IX] / X[:, self.HOUSEHOLD_IX]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.BEDROOMS_IX] / X[:, self.ROOMS_IX]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

