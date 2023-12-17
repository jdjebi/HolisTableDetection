import numpy as np


def show_coef(coef, features_names):
    coef_abs = abs(coef)
    features_index_desc = np.argsort(coef_abs)[::-1]
    lr_best_index = features_index_desc[:10]
    for index in lr_best_index:
        print(f' feature : {index} :', {features_names[index]})