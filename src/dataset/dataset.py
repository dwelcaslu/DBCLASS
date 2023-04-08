import matplotlib.pyplot as plt
import numpy as np
import random


# Basic configurations:
plt.rcParams.update({'font.size': 14, 'figure.figsize': (8, 6)})
np.random.seed(0)
random.seed(0)


def create_dataset(mean_array, sigma_array, n_samples=1000):
    """
    This definitions generates a random dataset based on
    gaussian distributions.
    """

    # Auxiliar variables:
    assert mean_array.shape[0] == sigma_array.shape[0], f"mean_array and sigma_array must be the same size"
    n_classes = mean_array.shape[0]
    n_features = mean_array.shape[1]
    samples_per_class = int(n_samples/n_classes)

    ds = {"data": None,
          "target": None,
          "target_names": np.array(["Class " + str(t + 1) for t in range(n_classes)]),
          "feature_names": np.array(["x{}".format(str(t + 1)) for t in range(n_features)])}

    # Innitializing X and y arrays:
    X = []
    y = []
    for c_index in range(n_classes):
        class_feats = []
        for feat_index in range(n_features):
            mu = mean_array[c_index][feat_index]
            sigma = sigma_array[c_index][feat_index]
            class_feats.append(np.random.normal(mu, sigma, samples_per_class))
        class_feats = np.array(class_feats)
        # Concatenating the features array:
        if X == []:
            X = class_feats.transpose()
        else:
            X = np.concatenate((X, class_feats.transpose()), axis=0)
        y += [c_index for i in range(samples_per_class)]

    ds["data"] = X
    ds["target"] = np.array(y, dtype='int32')

    return ds


def plot_dataset(ds, feat_index=(0, 1), labeled=True, fig_name=None, test_points=[]):
    """
    This definition plots the dataset, labeled by class or not.
    """

    # Setting the figure name:
    if fig_name is not None:
        fig_name = "dataset view - " + fig_name
    elif labeled is True:
        fig_name = "dataset view - labeled"
    else:
        fig_name = "dataset view - unlabeled"

    # Splitting the dataset in classes or not:
    if labeled is True:
        data = split_classes(ds)
    else:
        data = {'data': ds['data']}

    # Figure adjustments:
    plt.close(fig_name)
    plt.figure(fig_name)

    class_keys_list = [key for key in data.keys()]
    class_keys_list.sort()

    # Dealing with the negative class:
    noclass_label = "Unknown"
    if noclass_label in class_keys_list:
        class_keys_list.remove(noclass_label)
        negative_class = True
    else:
        negative_class = False

    for class_key in class_keys_list:
        plt.scatter(data[class_key][:, feat_index[0]], data[class_key][:, feat_index[1]], alpha=0.25, label=class_key, zorder=5)
    if negative_class is True:
        plt.scatter(data[noclass_label][:, feat_index[0]], data[noclass_label][:, feat_index[1]], alpha=0.25, color='black', label=noclass_label, zorder=3)
    if len(test_points) > 0:
        plt.scatter(test_points[:, feat_index[0]], test_points[:, feat_index[1]], color='k', marker='x', label='test point', zorder=10)

    # Setting the feature names in the axis:
    if 'feature_names' in ds:
        plt.xlabel(ds['feature_names'][feat_index[0]])
        plt.ylabel(ds['feature_names'][feat_index[1]])
    else:
        plt.xlabel('x{}'.format(str(feat_index[0] + 1)))
        plt.ylabel('x{}'.format(str(feat_index[1] + 1)))
    # Adding the classes labels to the legend:
    if labeled is True:
        ncol = min([len(class_keys_list), 3])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=ncol,
               fancybox=True, shadow=True, prop={'size': 14})
            
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def split_classes(ds):
    """
    This definition splits the data set between the classes.
    """

    data = {}
    for target in np.unique(ds['target']):
        label = ds['target_names'][target]
        data[label] = ds['data'][np.where(ds['target'] == target)]

    return data


def split_data(ds, prop_train=0.6):
    """
    This definitions splits the data for trainning and test.
    """

    ds_train = ds.copy()
    ds_test = ds.copy()
    X_train, y_train, X_test, y_test = None, None, None, None

    for target in np.unique(ds['target']):
        target_indexes = np.where(ds['target'] == target)
        n_regs_train = int(prop_train*len(target_indexes[0]))

        if X_train is None:
            X_train = ds['data'][target_indexes][0:n_regs_train]
            y_train = ds['target'][target_indexes][0:n_regs_train]
            X_test = ds['data'][target_indexes][n_regs_train:]
            y_test = ds['target'][target_indexes][n_regs_train:]
        else:
            X_train = np.concatenate((X_train, ds['data'][target_indexes][0:n_regs_train]), axis=0)
            y_train = np.concatenate((y_train, ds['target'][target_indexes][0:n_regs_train]), axis=0)
            X_test = np.concatenate((X_test, ds['data'][target_indexes][n_regs_train:]), axis=0)
            y_test = np.concatenate((y_test, ds['target'][target_indexes][n_regs_train:]), axis=0)

    ds_train['data'] = X_train
    ds_train['target'] = y_train
    ds_test['data'] = X_test
    ds_test['target'] = y_test

    return ds_train, ds_test

def join_data(ds_list):
    """
    """
    ds = ds_list[0]
    for new_ds in ds_list[1:]:
        ds['data'] = np.concatenate((ds['data'], new_ds['data']))
        ds['target'] = np.concatenate((ds['target'], new_ds['target']))
    return ds

def insert_rand_noclass(ds, noise_type="uniform"):
    """
    """
    n_samples = ds['data'].shape[0]
    n_features = ds['data'].shape[1]
#    n_classes = len(np.unique(ds['target']))
#    n_rand_samples = int(n_samples/n_classes)
    n_rand_samples = int(0.5 * n_samples)

    if noise_type == "uniform":
        random_data = np.random.uniform(ds['data'].min(axis=0), ds['data'].max(axis=0), (n_rand_samples, n_features))
    elif noise_type == "10max":
        random_data = 10*ds['data'].max()*np.random.rand(n_rand_samples, n_features)
    else:
        random_data = np.random.uniform(ds['data'].min(), ds['data'].max(), (n_rand_samples, n_features))
    noclass_target = -np.ones(n_rand_samples, dtype=int)

    ds['data'] = np.concatenate((ds['data'], random_data))
    ds['target'] = np.concatenate((ds['target'], noclass_target))
    
    return ds
