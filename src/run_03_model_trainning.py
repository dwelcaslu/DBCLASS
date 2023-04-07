# Third-party libraries:
import numpy as np
# My libraries:
import dataset.dataset as dtset
import dbclass.dbclass as DBclass
from dbclass import dbclass_utils as db_utils


if __name__ == "__main__":
    """
    TODO: Implement this docstring.
    """

    # Defining the mean and sigma arrays:
    mean_array = np.array([[10, 10],
                           [5, 20],
                           [10, 30],
                           [50, 50],
                           [30, 20],
                           [50, 10]])
    sigma_array = np.array([[2, 1],
                            [2, 3],
                            [6, 3],
                            [8, 4],
                            [8, 8],
                            [6, 7]])

    # Creating the dataset:
    ds = dtset.create_dataset(mean_array, sigma_array, n_regs=10000)
    # Splitting the data for trainning and for test:
    ds_train, ds_test = dtset.split_data(ds, prop_train=0.8)

    # Trainning the model using cross validation in the trainning dataset:
    dbclass = DBclass.PGC()
    prob_thold_list = [0, 0.1, 0.2, 0.3] + list(np.arange(0.4, 0.6, 0.01)) + [0.6, 0.75, 0.8, 0.999]
    best_prob_thold, class_metrics = db_utils.cross_validation_trainning(dbclass, ds_train, ds_test, prob_thold_list)
    print("\nBest probability threshold value:", best_prob_thold)
    for metric in class_metrics.keys():
        print(metric, class_metrics[metric])
