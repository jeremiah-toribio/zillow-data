# My py files
import env
import os
import wrangle as w
# Load Datasets
from pydataset import data
# Ignore Warning
import warnings
warnings.filterwarnings("ignore")
# Array and Dataframes
import numpy as np
import pandas as pd
# Imputer
from sklearn.impute import SimpleImputer
# Evaluation: Visualization
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# Evaluation: Statistical Analysis
from scipy import stats
# Modeling: Preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler, RobustScaler
# Modeling: Feature Selection
from sklearn.feature_selection import SelectKBest, f_regression, RFE
# Modeling
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import LassoLars
from sklearn.linear_model import TweedieRegressor
    # Linear: Polynomial
from sklearn.preprocessing import PolynomialFeatures
# Modeling: Selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# Metrics
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score

###                ###
# #     Metrics    # #
###                ###

def find_baseline(train, y_train):
    '''
    Sets and returns baseline.
    '''
    baseline = y_train.mean()
    # baseline array serves as predictions for the baseline, to measure baseline RSME & R2
    baseline_array = np.repeat(baseline, len(train))
    return baseline, baseline_array

def regression_metrics(y_train, predictions):
    '''
    Calculates regression model metrics using SKLearn's RMSE and R2.

    Kwargs: y_train (df), predictions (df)
    '''
    
    rmse = mean_squared_error(y_train, predictions, squared = False)
    r2 = r2_score(y_train, predictions)
    return rmse, r2

def metrics_dataframe(model,RMSE,R2):
    '''
    Keep track and automatically append data to compare models.
    '''
    metrics_df = pd.DataFrame(data=[
            {
                'model':model,
                'rmse':RMSE,
                'r2':R2
            }
            ])
    return metrics_df

def save_metrics(df, model, RMSE, R2):
        df.loc[len(df)] = [model, RMSE, R2]
        return df

###                  ###
# #     Feature      # #
###    Selection     ###

def rfe(x_train, y_train, model, k=2):
    # create the rfe object, indicating the ML object (lr) and the number of features I want to end up with. 
    if model == 'ols':
        model = lr()
    elif model == 'lassolars':
        model = LassoLars()
    elif model == model.__contains__('glm'):
        if model == ('glm0'):
            model == TweedieRegressor(power=0)
        elif model == ('glm1'):
            model == TweedieRegressor(power=1)
        elif model == ('glm2'):
            model == TweedieRegressor(power=2)
        elif model == ('glm3'):
            model == TweedieRegressor('glm3')
    else:
        raise ValueError('Select a valid model.')
    
    rfe = RFE(model, n_features_to_select=k)

    # fit the data using RFE
    rfe.fit(x_train,y_train)  

    # get the mask of the columns selected
    feature_mask = rfe.support_

    # get list of the column names. 
    rfe_feature = x_train.iloc[:,feature_mask].columns.tolist()

    return rfe_feature

def select_kbest(x_train, y_train, k=2):

    # parameters: f_regression stats test, give me all features - normally in
    f_selector = SelectKBest(f_regression, k=k)#k='all')
    # find the all X's correlated with y
    f_selector.fit(x_train, y_train)

    # boolean mask of whether the column was selected or not
    feature_mask = f_selector.get_support()

    # get list of top K features. 
    f_feature = x_train.iloc[:,feature_mask].columns.tolist()

    return f_feature

###              ###
# #    Models    # #
###              ###

def OLS(x_train, y_train, x_validate):
    '''
    Also returns validate set predictions in second return.
    '''

    lr1 = lr()
    lr1.fit(x_train, y_train)
    predictions = lr1.predict(x_train)
    val_predictions = lr1.predict(x_validate)
    rmse, r2 = regression_metrics(y_train, predictions)
    intercept = lr1.intercept_
    coef = lr1.coef_[0]
    print(f'RMSE: {rmse}')
    print(f'R2: {r2}')
    print(f'Intercept: {intercept}')
    print(f'Coefficient: {coef}')
    return predictions, val_predictions

def Lars(x_train, y_train, x_validate):
    '''
    Also returns validate set predictions in second return.
    '''

    lars = LassoLars(alpha=0)
    lars.fit(x_train, y_train)
    predictions = lars.predict(x_train)
    val_predictions = lars.predict(x_validate)
    rmse, r2 = regression_metrics(y_train, predictions)
    intercept = lars.intercept_
    coef = lars.coef_[0]
    print(f'RMSE: {rmse}')
    print(f'R2: {r2}')
    print(f'Intercept: {intercept}')
    print(f'Coefficient: {coef}')   
    return predictions, val_predictions

def Polynom(x_train, y_train, x_validate):
    '''
    Also returns validate set predictions in second return.
    '''

    pr = lr()
    pr.fit(x_train, y_train)
    predictions = pr.predict(x_train)
    val_predictions = pr.predict(x_validate)
    rmse, r2 = regression_metrics(y_train, predictions)
    intercept = pr.intercept_
    coef = pr.coef_[0]
    print(f'RMSE: {rmse}')
    print(f'R2: {r2}')
    print(f'Intercept: {intercept}')
    print(f'Coefficient: {coef}')   
    return predictions, val_predictions

def GLM(x_train, y_train, x_validate,  power=2):
    '''
    Also returns validate set predictions in second return.
    '''

    glm = TweedieRegressor(power = power)
    glm.fit(x_train, y_train)
    predictions = glm.predict(x_train)
    val_predictions = glm.predicted(x_validate)
    rmse, r2 = regression_metrics(y_train, predictions)
    intercept = glm.intercept_
    coef = glm.coef_[0]
    print(f'RMSE: {rmse}')
    print(f'R2: {r2}')
    print(f'Intercept: {intercept}')
    print(f'Coefficient: {coef}')   
    return predictions, val_predictions