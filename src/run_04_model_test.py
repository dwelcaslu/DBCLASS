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

    # Once the best probability score threshold is found, the classifier is
    # configured with the best settings:
    dbclass = DBclass.PGC(prob_thold=0.5)
    dbclass.fit(ds)
    db_utils.dbclass_model_test(dbclass, ds)
