from django.shortcuts import render
# ---------------------------------
from ..core.Chapter_01 import Algo1
from ..core.Chapter_02 import Algo2, CombinedAttributesAdder
# ---------------------------------
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
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
from sklearn.svm import SVR
from sklearn import linear_model, neighbors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing
from sklearn import pipeline
from scipy.stats import randint
from scipy import stats
# ---------------------------------


def index(request):
    return render(request, 'introml/index.html', {'x': 500})


# Main Navigator ---
def show_content(request):
    chapter=request.POST.get('chapter')
    page=request.POST.get('page')
    # the variable chapters send me to the right chapter function --
    # the variable page send me to the right section in the function function (see below) --
    # --- chapter 1 --
    if str(chapter) == '1':
        return chapter_1(request, page)
    # if str(chapter) == '1_1':
    #     return chapter_1_1(request, page)
    # --- chapter 2 --
    elif str(chapter) == '2':
        return chapter_2(request, page)
    elif str(chapter) == '2_1':
        return chapter_2_1(request, page)
    elif str(chapter) == '2_2':
        return chapter_2_2(request, page)
    # --- chapter 3 --


#  -- Every Chapter has it own code --
# -------------------------------------------------------------------
def ch01(request):
    title = 'The Machine Learning Landscape'
    return render(request, 'introml/ch01.html', {'title': title})


def chapter_1(request, page):
    algo = Algo1()
    title = 'GDP/C vs Life satisfaction'
    if page == "plot_gdp_pc_vs_life_satisfaction":
        title = 'GDP/C vs Life satisfaction'
        algo.sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
        fig_id = "plot_gdp_pc_vs_life_satisfaction"
        algo.save_fig(fig_id)
        return render(request, 'introml/_show_plot.html',
                      {'title': title, 'fig_id': fig_id, 'chapter_id': algo.CHAPTER_ID, 'img_type':'png'})
    elif page == "linear_model":
        title = 'Select a linear model for Prediction'
        X = np.c_[algo.sample_data["GDP per capita"]]
        y = np.c_[algo.sample_data["Life satisfaction"]]
        model = linear_model.LinearRegression()  # linear reg
        # model = neighbors.KNeighborsRegressor(n_neighbors=3)   #  k-neighbors
        # Train the model
        model.fit(X, y)
        # Make a prediction for Cyprus
        X_new = [[22587]]  # Cyprus' GDP per capita
        print('-'*10)
        print(title)
        print('--')
        print('X_new = ' + str(X_new))
        print('-'*10)
        y = model.predict(X_new)
        print('output:  y = ' + str(y))  # outputs [[ 5.96242338]]
        print('-'*10)
        return render(request, 'introml/_data_processed_successfully.html', {'title': title})
    elif page == "several_countries_on_plot":
        title = 'Several countries on plot'
        algo.sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
        plt.axis([0, 60000, 0, 10])
        position_text = {
            "Hungary": (5000, 1),
            "Korea": (18000, 1.7),
            "France": (29000, 2.4),
            "Australia": (40000, 3.0),
            "United States": (52000, 3.8),
        }
        for country, pos_text in position_text.items():
            pos_data_x, pos_data_y = algo.sample_data.loc[country]
            country = "U.S." if country == "United States" else country
            plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
                         arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
            plt.plot(pos_data_x, pos_data_y, "ro")
        fig_id = "money_happy_scatterplot"
        algo.save_fig(fig_id)
        # plt.show()
        saved_data = "lifesat.csv"
        algo.sample_data.to_csv(os.path.join(algo.TO_DATA_PATH, saved_data))
        print('sample_data')
        print(algo.sample_data.loc[list(position_text.keys())])
        return render(request, 'introml/ch01/_several_countries_on_plot.html',
                      {'title': title, 'fig_id': fig_id, 'chapter_id': algo.CHAPTER_ID,
                       'img_type': 'png', 'saved_data': saved_data})
    elif page == "tweaking_model_params_plot":
        title = 'GDP/C vs Life satisfaction (diff params)'
        algo.sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
        plt.axis([0, 60000, 0, 10])
        X = np.linspace(0, 60000, 1000)
        plt.plot(X, 2 * X / 100000, "r")
        plt.text(40000, 2.7, r"$\theta_0 = 0$", fontsize=14, color="r")
        plt.text(40000, 1.8, r"$\theta_1 = 2 \times 10^{-5}$", fontsize=14, color="r")
        plt.plot(X, 8 - 5 * X / 100000, "g")
        plt.text(5000, 9.1, r"$\theta_0 = 8$", fontsize=14, color="g")
        plt.text(5000, 8.2, r"$\theta_1 = -5 \times 10^{-5}$", fontsize=14, color="g")
        plt.plot(X, 4 + 5 * X / 100000, "b")
        plt.text(5000, 3.5, r"$\theta_0 = 4$", fontsize=14, color="b")
        plt.text(5000, 2.6, r"$\theta_1 = 5 \times 10^{-5}$", fontsize=14, color="b")
        fig_id1 = 'tweaking_model_params_plot'
        fig_name_id1 = 'Tweaking model params plot'
        algo.save_fig(fig_id1)
        # plt.show()
        lin1 = linear_model.LinearRegression()
        Xsample = np.c_[algo.sample_data["GDP per capita"]]
        ysample = np.c_[algo.sample_data["Life satisfaction"]]
        lin1.fit(Xsample, ysample)
        t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]
        print(t0, t1)
        algo.sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
        plt.axis([0, 60000, 0, 10])
        X = np.linspace(0, 60000, 1000)
        plt.plot(X, t0 + t1*X, "b")
        fig_id2 = 'regression_line'
        algo.save_fig(fig_id2)
        fig_name_id2 = 'Regression Line'
        return render(request, 'introml/ch01/_tweaking_model_params_plot.html',
                      {'title': title,
                       'fig_id1': fig_id1, 'fig_name_id1': fig_name_id1,
                       'fig_id2': fig_id2, 'fig_name_id2': fig_name_id2,
                       'chapter_id': algo.CHAPTER_ID, 'img_type': 'png'})
    elif page == "tweaking_model_params_plot":
        title = 'GDP/C vs Life satisfaction (diff params)'
        return render(request, 'introml/ch01/_tweaking_model_params_plot.html',
                      {'title': title,
                       'fig_id1': fig_id1, 'fig_name_id1': fig_name_id1,
                       'fig_id2': fig_id2, 'fig_name_id2': fig_name_id2,
                       'chapter_id': algo.CHAPTER_ID, 'img_type': 'png'})

# -------------------------------------------------------------------
def ch02(request):
    title = 'End-to-End Machine Learning Project'
    return render(request, 'introml/ch02.html', {'title': title})


def chapter_2(request, page):
    algo = Algo2()
    if page == "get_data":
        title = "Got the Data"
        housing = algo.DATA
        print(housing.head(20))
        print(housing.info())
        print('-'*10)
        print(housing["ocean_proximity"].value_counts())
        print('-'*10)
        print(housing.describe())
        return render(request, 'introml/ch02/_get_data.html', {'title': title, 'df': algo.DATA.head(20)})
    elif page == 'attribute_histogram_plots':
        title = "Histogram of Attributes"
        algo.DATA.hist(bins=50, figsize=(20, 15))
        fig_id = "attribute_histogram_plots"
        algo.save_fig(fig_id)
        return render(request, 'introml/ch02/_attribute_histogram_plots.html',
                      {'title': title, 'fig_id': fig_id, 'chapter_id': algo.CHAPTER_ID})
    elif page == "california_housing_prices_scatter_plot":
        fig_id = "housing_prices_scatter_plot"
        title = "California housing prices scatter plot"
        algo.train_test_stratified_split(test_size=0.2, field="median_income",
                                         bins=[0., 1.5, 3.0, 4.5, 6., np.inf], lables=[1, 2, 3, 4, 5])
        algo.plot(x="longitude", y="latitude", s="population", snf=100,
                  c="median_house_value", fig_name=fig_id, img_name="california", img_type="png")
        return render(request, 'introml/ch02/_california_housing_prices_scatter_plot.html',
                      {'title': title, 'fig_id': "california_"+fig_id, 'chapter_id': algo.CHAPTER_ID, 'img_type': 'png'})
    elif page == "data_exploration":
        # ----- Data Exploration -----
        algo.train_test_stratified_split(test_size=0.2, field="median_income",
                                         bins=[0., 1.5, 3.0, 4.5, 6., np.inf], lables=[1, 2, 3, 4, 5])
        title = 'Data Exploration'
        train_data = algo.TRAIN.copy()
        corr_matrix = train_data.corr()
        cm = corr_matrix[algo.TARGET_FIELD].sort_values(ascending=False)
        print(cm)
        attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
        scatter_matrix(train_data[attributes], figsize=(12, 8))
        fig_id_1 = "scatter_matrix_plot"
        fig_name_1 = "Scatter matrix plot"
        algo.save_fig(fig_id_1)
        # plt.show()
        # --
        train_data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
        plt.axis([0, 16, 0, 550000])
        fig_id_2 = "income_vs_house_value_scatter_plot"
        fig_name_2 = "Income vs house value scatter plot"
        algo.save_fig(fig_id_2)

        train_data["rooms_per_household"] = train_data["total_rooms"] / train_data["households"]
        train_data["bedrooms_per_room"] = train_data["total_bedrooms"] / train_data["total_rooms"]
        train_data["population_per_household"] = train_data["population"] / train_data["households"]

        corr_matrix = train_data.corr()
        corr_target = corr_matrix[algo.TARGET_FIELD].sort_values(ascending=False)
        print(corr_target)

        train_data.plot(kind="scatter", x="rooms_per_household", y="median_house_value", alpha=0.2)
        plt.axis([0, 5, 0, 520000])
        fig_id_3 = "rooms_per_household_to_median_house_value"
        fig_name_3 = "Rooms per household vs median house value"
        algo.save_fig(fig_id_3)
        # plt.show()
        print(train_data.describe())
        # ----- End Data Exploration -----
        return render(request, 'introml/ch02/_data_exploration.html',
                      {'title': title, 'chapter_id': algo.CHAPTER_ID,
                       'fig_id_1': fig_id_1, 'fig_name_1': fig_name_1,
                       'fig_id_2': fig_id_2, 'fig_name_2': fig_name_2,
                       'fig_id_3': fig_id_3, 'fig_name_3': fig_name_3,
                       'img_type': 'png'})


def chapter_2_1(request, page):
    algo = Algo2()
    algo.train_test_stratified_split(test_size=0.2, field="median_income",
                                     bins=[0., 1.5, 3.0, 4.5, 6., np.inf], lables=[1, 2, 3, 4, 5])
    # Prepare the data for Machine Learning algorithms
    print('Prepare the data for Machine Learning algorithms')
    algo.set_target_data()
    if page == "prepare_the_data_for_ml":
        title = "Prepare the data for ML"
        # Prepare the Data for Machine Learning Algorithms
        print('Prepare the Data for Machine Learning Algorithms')
        # ---- Data Cleaning ---
        print('Data Cleaning')
        print('-'*20)
        housing = algo.TRAIN_DATA.copy()
        sample_incomplete_rows = housing[algo.TRAIN_DATA.isnull().any(axis=1)].head()
        print(sample_incomplete_rows)
        print('-'*10)
        print('sample_incomplete_rows["total_bedrooms"]')
        print('-'*10)
        print(sample_incomplete_rows["total_bedrooms"])
        print('-'*10)
        print('In the code I commented out how to delete rows or columns with missing data')
        # sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1
        # sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2
        print('-'*10)
        print('Fill missing data with median')
        print('-'*30)
        median = housing["total_bedrooms"].median()
        sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
        print(sample_incomplete_rows["total_bedrooms"])
        print('-'*20)
        # --
        print('Use SimpleImputer')
        print('-'*30)
        imputer = SimpleImputer(strategy="median")
        # Remove the text attribute because median can only be calculated on numerical attributes:
        housing_num = housing.drop('ocean_proximity', axis=1)
        imputer.fit(housing_num)
        print('See that this is the same as manually computing the median of each attribute:')
        print('-'*10)
        print(imputer.statistics_)
        print('-'*10)
        # Check that this is the same as manually computing the median of each attribute:
        print(housing_num.median().values)
        print('-'*20)

        # Transform the training set:
        X = imputer.transform(housing_num)
        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=algo.TRAIN_DATA.index)
        print('housing_tr.head()')
        print(housing_tr.head())
        print('housing_tr.loc[sample_incomplete_rows.index.values]["total_bedrooms"]')
        print(housing_tr.loc[sample_incomplete_rows.index.values]["total_bedrooms"])
        print('imputer.strategy')
        print(imputer.strategy)
        print('-'*30)

        # -- Handling Text and Categorical Attributes --
        print('Handling Text and Categorical Attributes')
        print("Now let's pre-process the categorical input feature, `ocean_proximity`:")
        print('Use the OneHotEncoder:')
        print('-'*30)
        housing_cat = algo.TRAIN_DATA[['ocean_proximity']]
        cat_encoder = OneHotEncoder(sparse=False)
        housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
        print('housing_cat_1hot')
        print(housing_cat_1hot)
        # ---- End Data Cleaning ---
        return render(request, 'introml/ch02/_data_processed_successfully.html',
                      {'title': title})
    elif page == "custom_transformers":
        title = "Custom Transformers"
        # --
        attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False, algo=algo)
        housing_extra_attribs = attr_adder.fit_transform(algo.TRAIN_DATA.values)
        print('housing_extra_attribs')
        print('-'*20)
        print(housing_extra_attribs)
        print('-'*10)
        housing_extra_attribs = pd.DataFrame(
            housing_extra_attribs,
            columns=list(algo.TRAIN_DATA.columns)+["rooms_per_household", "population_per_household"],
            index=algo.TRAIN_DATA.index)
        print('housing_extra_attribs.head()')
        print('-'*20)
        print(housing_extra_attribs.head())
        # --- End Custom Transformers ---
        return render(request, 'introml/ch02/_data_processed_successfully.html',
                      {'title': title})


def chapter_2_2(request, page):
    algo = Algo2()
    algo.train_test_stratified_split(test_size=0.2, field="median_income",
                                     bins=[0., 1.5, 3.0, 4.5, 6., np.inf], lables=[1, 2, 3, 4, 5])
    algo.set_target_data()
    # --
    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder',  CombinedAttributesAdder(algo=algo)),
            ('std_scaler', StandardScaler()),
        ])
    housing_num = algo.TRAIN_DATA.drop('ocean_proximity', axis=1)
    algo.num_attribs = list(housing_num)
    print(algo.num_attribs)
    algo.cat_attribs = ["ocean_proximity"]
    algo.extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    data_pipeline = ColumnTransformer([
            ("num", num_pipeline, algo.num_attribs),
            ("cat", OneHotEncoder(), algo.cat_attribs),
        ])
    algo.set_pipeline(data_pipeline)
    if page == "transformation_pipelines_and_scaling":
        title = "Transformation Pipelines & Scaling"
        housing_num_tr = num_pipeline.fit_transform(housing_num)
        print(title)
        print('housing_num_tr')
        print('-'*10)
        print(housing_num_tr)
        print('data_pipeline')
        print('-'*10)
        print('algo.TRAIN_DATA')
        print(algo.TRAIN_DATA)
        print('-'*10)
        print('algo.TRAIN_DATA.shape')
        print(algo.TRAIN_DATA.shape)
        return render(request, 'introml/ch02/_data_processed_successfully.html', {'title': title})
    elif page == "select_and_train_model":
        title = "Select and Train a Model"
        print(title)
        print('---')
        print('Select and Train a Model')
        print('Training and Evaluating on the Training Set')
        print('lin_reg')
        print('-'*30)
        lin_reg = LinearRegression()
        lin_reg.fit(algo.train_data, algo.TRAIN_TARGET)
        print("let's try the full pre-processing pipeline on a few training instances")
        print('-'*20)
        some_data = algo.TRAIN_DATA.iloc[:5]
        some_labels = algo.TRAIN_TARGET.iloc[:5]
        some_data_prepared = data_pipeline.transform(some_data)
        print("Predictions:", lin_reg.predict(some_data_prepared))
        print('Compare against the actual values:')
        print('-'*10)
        print("Labels:", list(some_labels))
        print(some_data_prepared)
        print('-'*20)

        train_predictions = lin_reg.predict(algo.train_data)
        lin_mse = mean_squared_error(algo.TRAIN_TARGET, train_predictions)
        lin_rmse = np.sqrt(lin_mse)
        print('lin_rmse')
        print(lin_rmse)
        print('-'*20)
        print('Use the object to run models: Not Using Cross-Validation')
        print('-'*30)
        lin_rmse, lin_mae = algo.run_model(model=LinearRegression(), name="linear")
        print("LinearRegression: rmse:  ", lin_rmse, "mae: ", lin_mae)
        print('-'*10)
        tree_reg_rmse, tree_reg_mae = algo.run_model(model=DecisionTreeRegressor(random_state=algo.RANDOM_STATE), name="tree_reg")
        print("DecisionTreeRegressor: rmse: ", tree_reg_rmse, "mae: ", tree_reg_mae)
        print('-'*10)
        forest_reg_rmse, forest_reg_mae = \
            algo.run_model(model=RandomForestRegressor(n_estimators=10, random_state=algo.RANDOM_STATE), name="forest_reg")
        print("RandomForestRegressor: rmse:  ", forest_reg_rmse, "mae: ", forest_reg_mae)
        print('-'*10)
        print('Use the object to run models: Better Evaluation Using Cross-Validation')
        print('-'*30)
        scores = algo.run_model_cv(model=LinearRegression(), cv=10, name="linearCV")
        print("\nLinearRegression:  \n", "Scores:", scores, "Mean:", scores.mean(), "Standard deviation:", scores.std())

        scores = algo.run_model_cv(model=DecisionTreeRegressor(random_state=algo.RANDOM_STATE), cv=10, name="tree_regCV")
        print("\nDecisionTreeRegressor:  \n", "Scores:", scores, "Mean:", scores.mean(), "Standard deviation:", scores.std())
        # --
        scores = algo.run_model_cv(model=RandomForestRegressor(n_estimators=10, random_state=algo.RANDOM_STATE), cv=10,
                                   name="forest_regCV")
        print("\nRandomForestRegressor:  \n", "Scores:", scores, "Mean:", scores.mean(), "Standard deviation:", scores.std())
        # -- It takes a long time --
        # scores = algo.run_model_cv(model=SVR(kernel="linear"), cv=10, name="svm_regCV")
        # print("SVR:  \n\n", "Scores:", scores, "Mean:", scores.mean(), "Standard deviation:", scores.std())
        return render(request, 'introml/ch02/_data_processed_successfully.html', {'title': title})
    elif page == "fine-tune_your_model_grid_search":
        title = "Fine - Tune Your Model: Grid Search"
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]
        forest_reg = RandomForestRegressor(algo.RANDOM_STATE)
        # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                                   scoring='neg_mean_squared_error', return_train_score=True)
        # grid_search.fit(algo.train_data, algo.TRAIN_TARGET)
        df_grid_search, sorted_feature_importances, rmse, confidence_interval = \
            algo.run_model_grid_search_cv(grid_search=grid_search, model_name='GridSearchCV_forest_reg')
        return render(request, 'introml/ch02/_fine-tune_your_model_grid_search.html',
                      {'title': title, 'df_grid_search': df_grid_search})
    elif page == "predictions_with_saved_model":
        title = "Predictions with saved model"
        rmse, confidence_interval = algo.prediction_and_accuracy(model_name='GridSearchCV_forest_reg')
        return render(request, 'introml/ch02/_predictions_with_saved_model.html',
                      {'title': title, 'rmse': rmse, 'confidence_interval': confidence_interval})
    elif page == "fine-tune_your_model_randomized_search":
        title = 'Randomized Grid Search CV'
        print(title)
        print('-'*30)
        param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }
        forest_reg = RandomForestRegressor(random_state=algo.RANDOM_STATE)
        rnd_grid_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                             n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)

        df_grid_search, sorted_feature_importances, rmse, confidence_interval = \
            algo.run_model_grid_search_cv(grid_search=rnd_grid_search, model_name='RandomizedSearchCV_forest_reg')
        print(df_grid_search, sorted_feature_importances, rmse, confidence_interval)

        # rnd_search.fit(algo.train_data, algo.TRAIN_TARGET)
        # cvres = rnd_search.cv_results_
        # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        #     print(np.sqrt(-mean_score), params)
        # return render(request, 'introml/ch02/_data_processed_successfully.html',
        #               {'title': title})
        return render(request, 'introml/ch02/_fine-tune_your_model_grid_search.html',
                      {'title': title, 'df_grid_search': df_grid_search})


# -------------------------------------------------------------------
def ch03(request):
    title = 'Ch03: Classification'
    return render(request, 'introml/ch03.html', {'title': title})

