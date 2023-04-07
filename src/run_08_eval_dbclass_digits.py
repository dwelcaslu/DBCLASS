# Third-party libraries:
import numpy as np
from sklearn.datasets import load_digits
# My libraries:
import dataset.dataset as dtset
import dbclass.dbclass as DBclass
from dbclass import dbclass_utils as db_utils


if __name__ == "__main__":
    """
    TODO: Implement this docstring.
    """

    # import some data to play with
    ds = load_digits()
    ds['data'] = db_utils.get_pca_data(ds['data'], n_components=15)
    # Splitting the data for trainning and for test:
    ds_train, ds_validation = dtset.split_data(ds, prop_train=0.5)
    ds_validation, ds_test = dtset.split_data(ds_validation, prop_train=0.5)

    # Trainning the model using cross validation in the trainning dataset:
    dbclass = DBclass.PGC()
    prob_thold_list = [0, 0.05, 0.1, 0.15, 0.25, 0.3, 0.35] + list(np.arange(0.4, 0.6, 0.01)) + [0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.999]
    best_prob_thold, class_metrics = db_utils.cross_validation_trainning(dbclass, ds_train, ds_validation, prob_thold_list)
    print("\nBest probability threshold value:", best_prob_thold)
    for metric in class_metrics.keys():
        print(metric, class_metrics[metric])

    # Once the best probability score threshold is found, the classifier is
    # configured with the best settings:
    dbclass = DBclass.PGC(prob_thold=best_prob_thold)
    ds_model = dtset.join_data((ds_train, ds_validation))
    dbclass.fit(ds_model)
    ds_test = dtset.insert_rand_noclass(ds_test)
    db_utils.dbclass_model_test(dbclass, ds_test)
