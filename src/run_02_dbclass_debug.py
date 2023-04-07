# Third-party libraries:
import numpy as np
# My libraries:
import dataset.dataset as dtset
import dbclass.dbclass as dbclass


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

    # Using fit and predict:
    dbclass = dbclass.PGC()
    dbclass.fit(ds)

    # Defining some test points:
    test_points = np.array([[10, 10],
                           [5, 20],
                           [10, 30],
                           [50, 50],
                           [30, 20],
                           [50, 10],
                           [0, 0],
                           [6, 12],
                           [13, 15],
                           [4, 26],
                           [21, 30],
                           [40, 16]])

    prob_thold_list = [0, 0.2, 0.5, 0.8]
    for prob_thold in prob_thold_list:
        print("\nProbability score threshold:", prob_thold)
        y_predictions, y_scores = dbclass.predict(test_points, prob_thold=prob_thold)
        for point, y_predict, score in zip(test_points, y_predictions, y_scores):
            print(point, y_predict, round(score, 6))

    # Plotting the dataset with the test points:
    dtset.plot_dataset(ds, feat_index=(0, 1), labeled=True, test_points=test_points)
