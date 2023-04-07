# Third-party libraries:
import numpy as np
from sklearn.datasets import load_wine
# My libraries:
import dataset.dataset as dtset
import dbclass.dbclass as DBclass
from dbclass import dbclass_utils as db_utils


if __name__ == "__main__":
    """
    TODO: Implement this docstring.
    """
    
    # import some data to play with
    ds = load_wine()
    # Splitting the data for trainning and for test:
    ds_train, ds_test = dtset.split_data(ds, prop_train=0.8)

    # Trainning the model using cross validation in the trainning dataset:
    dbclass = DBclass.PGC()
    prob_thold_list = np.arange(0, 1, 0.01)
    best_prob_thold, class_metrics = db_utils.cross_validation_trainning(dbclass, ds_train, ds_test, prob_thold_list)
    print("\nBest probability threshold value:", best_prob_thold)
    for metric in class_metrics.keys():
        print(metric, class_metrics[metric])

    # Once the best probability score threshold is found, the classifier is
    # configured with the best settings:
    dbclass = DBclass.PGC(prob_thold=best_prob_thold)
    dbclass.fit(ds)
    db_utils.dbclass_model_test(dbclass, ds)
