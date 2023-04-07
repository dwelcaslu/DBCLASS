# Third-party libraries:
from sklearn.datasets import fetch_lfw_people
# My libraries:
from dbclass import dbclass_utils as db_utils


if __name__ == "__main__":
    """
    TODO: Implement this docstring.
    """

    # import some data to play with
    ds = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_components_best, max_confidence = db_utils.perform_pca_test(ds)
    print('Best number of components:', n_components_best)
    print('Confidence level:', max_confidence)
