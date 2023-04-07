import dataset.dataset as dtset
import numpy as np

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

    # Plotting the dataset:
    dtset.plot_dataset(ds, feat_index=(0, 1), labeled=False)
    dtset.plot_dataset(ds, feat_index=(0, 1), labeled=True)

    # Splitting the data for trainning and for test:
    ds_train, ds_test = dtset.split_data(ds, prop_train=0.7)
    dtset.plot_dataset(ds_train, feat_index=(0, 1), labeled=True, fig_name="train")
    dtset.plot_dataset(ds_test, feat_index=(0, 1), labeled=True, fig_name="test")
