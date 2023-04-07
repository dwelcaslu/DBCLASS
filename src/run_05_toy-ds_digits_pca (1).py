# Third-party libraries:
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
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

    # Splitting the data for trainning and for test:
    ds_train, ds_test = dtset.split_data(ds, prop_train=0.8)

    # Compute a PCA on the face dataset (treated as unlabeled dataset):
    # unsupervised feature extraction / dimensionality reduction
    n_components = 10
    pca = PCA(n_components=n_components, svd_solver='full',
              whiten=True).fit(ds_train['data'])
    # Projecting the input data on the eigenfaces orthonormal basis:
    ds_train['data'] = pca.transform(ds_train['data'])
    ds_test['data'] = pca.transform(ds_test['data'])
    ds['data'] = pca.transform(ds['data'])

    # Trainning the model using cross validation in the trainning dataset:
    dbclass = DBclass.PGC()
    best_prob_thold, class_metrics = db_utils.cross_validation_trainning(dbclass, ds_train, ds_test)
    print("\nBest probability threshold value:", best_prob_thold)
    for metric in class_metrics.keys():
        print(metric, class_metrics[metric])

    # Once the best probability score threshold is found, the classifier is
    # configured with the best settings:
    dbclass = DBclass.PGC(prob_thold=best_prob_thold)
    dbclass.fit(ds)
    db_utils.dbclass_model_test(dbclass, ds)
