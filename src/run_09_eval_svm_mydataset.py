# Third-party libraries:
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# My libraries:
import dataset.dataset as dtset
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

    # import some data to play with
    ds = dtset.create_dataset(mean_array, sigma_array, n_regs=10000)
    # Splitting the data for trainning and for test:
    ds_train, ds_validation = dtset.split_data(ds, prop_train=0.5)
    ds_validation, ds_test = dtset.split_data(ds_validation, prop_train=0.5)
    ds_train = dtset.join_data((ds_train, ds_validation))

    # Train a SVM classification model
    # Fitting the classifier to the training set:
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                       param_grid, cv=5, iid=False)
    clf = clf.fit(ds_train['data'], ds_train['target'])
    print("\nBest estimator found by grid search:")
    print(clf.best_estimator_)

    # Quantitative evaluation of the model quality on the test set:
    ds_test = dtset.insert_rand_noclass(ds_test)
    y_pred = clf.predict(ds_test['data'])
    ds_pred = db_utils.print_classification_report(ds_test, y_pred)
    confusion_matrix = db_utils.build_confusion_matrix(ds_test['target'], y_pred)
    print('\nConfusion matrix:')
    print(confusion_matrix)
    class_metrics = db_utils.get_class_metrics(confusion_matrix)
    for metric in class_metrics.keys():
        print(metric, class_metrics[metric])
    dtset.plot_dataset(ds_pred, feat_index=(0, 1), labeled=True, fig_name="model test")
