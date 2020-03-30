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


class Algo1(Algo):
    def __init__(self):
        super().__init__(chapter_id="CH01_fundamentals", to_data_path="lifesat", target_field=None)
        oecd_bli = pd.read_csv(self.TO_DATA_PATH + "/oecd_bli_2015.csv", thousands=',')
        gdp_per_capita = pd.read_csv(self.TO_DATA_PATH + "/gdp_per_capita.csv", thousands=',', delimiter='\t',
                                     encoding='latin1', na_values="n/a")
        self.full_country_stats, self.sample_data, self.missing_data = self.prepare_country_stats(oecd_bli, gdp_per_capita)

    def prepare_country_stats(self, oecd_bli, gdp_per_capita):
        oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
        # print(oecd_bli.head())
        oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
        # print(oecd_bli.head())
        gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
        gdp_per_capita.set_index("Country", inplace=True)
        # print(gdp_per_capita.head())
        full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                      left_index=True, right_index=True)
        full_country_stats.sort_values(by="GDP per capita", inplace=True)
        # print(full_country_stats.head())
        remove_indices = [0, 1, 6, 8, 33, 34, 35]
        keep_indices = list(set(range(36)) - set(remove_indices))
        return full_country_stats[["GDP per capita", 'Life satisfaction']], \
               full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices], \
               full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[remove_indices]


