# Third-party libraries:
from sklearn.datasets import load_digits
# My libraries:
from dbclass import dbclass_utils as db_utils


if __name__ == "__main__":
    """
    TODO: Implement this docstring.
    """

    # import some data to play with
    ds = load_digits()
    n_components_best, max_confidence = db_utils.perform_pca_test(ds)
    print('\nBest number of components:', n_components_best)
    print('Confidence level:', max_confidence)
