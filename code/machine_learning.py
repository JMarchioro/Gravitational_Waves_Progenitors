import pandas as pd
import seaborn as sns
import math
import scipy
import numpy as np
import itertools

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

def score(model, data, target):
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    scoring = ["r2", "neg_mean_absolute_error", "neg_mean_squared_error",
               "neg_median_absolute_error", "max_error"]
    cv_results = cross_validate(model, data, target, scoring=scoring, cv=cv)
    scores = {"R2": cv_results["test_r2"],
              "MAE": -cv_results["test_neg_mean_absolute_error"],
              "MSE": -cv_results["test_neg_mean_squared_error"],
              "MedAE": -cv_results["test_neg_median_absolute_error"],
              "max": -cv_results["test_max_error"]}
    scores = pd.DataFrame(scores)
    return scores


def split(data, target):
    data_train, data_test, target_train, target_test = train_test_split(
    data, target, train_size=0.8, random_state=42)
    return data_train, data_test, target_train, target_test


def eval_model(model, data_train, data_test, target_train, target_test):
    y_keys = target_train.columns
    for key in y_keys:
        print(key)
        print("".join(50*['=']))
        model.fit(data_train.values, target_train[key])
        y_pred = model.predict(data_test)
        target_test[key+"_pred"] = y_pred
        scoring = score(model, data_train, target_train[key])
        print(f" R2 score : {scoring['R2'].mean():.3f} ± {scoring['R2'].std():.3f}")
        print(f" Mean absolute error score : {scoring['MAE'].mean():.3f} ± {scoring['MAE'].std():.3f}")
        print(f" Mean squared error score : {scoring['MSE'].mean():.3f} ± {scoring['MSE'].std():.3f}")
        print(f" Median absolute error score : {scoring['MedAE'].mean():.3f} ± {scoring['MedAE'].std():.3f}")
        print(f" max error score : {scoring['max'].mean():.3f} ± {scoring['max'].std():.3f}")
        print("\n")
    return data_test, target_test

def main(file_data, file_target):
    data = pd.read_hdf(file_data)
    target = pd.read_hdf(file_target)
    model = make_pipeline(RandomForestRegressor(random_state=0, n_estimators=10))
    data_train, data_test, target_train, target_test = split(data, target)
    data_test, target_test_fin = eval_model(model, data_train, data_test, target_train, target_test)
    return data_test, target_test_fin, model

# file_data = '/work/marchior/MCMC/data/data.h5'
# file_target = '/work/marchior/MCMC/data/target.h5'
