# Third-party libraries:
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
# My libraries:
import dataset.dataset as dtset
from dbclass.dbclass import DBCLASS


# Basic configurations:
plt.rcParams.update({'font.size': 14, 'figure.figsize': (8, 6)})


def cross_validation_trainning(dbclass, ds_train, ds_test, prob_thold_list=[], pca_test=False):
    """
    """

    if prob_thold_list == []:
        prob_thold_list = np.append(np.arange(0, 1, 0.1), 0.999)

    # Metrics arrays:
    accuracy_vec = np.zeros(len(prob_thold_list))
    prob_score_vec = np.zeros(len(prob_thold_list))
    confidence_vec = np.zeros(len(prob_thold_list))
    rejection_vec = np.zeros(len(prob_thold_list))
    # Optimum settings:
    max_train_confidence = 0
    best_prob_thold = 0
    best_metrics = {}

    dbclass.fit(ds_train['data'], ds_train['target'])
    for t_index, prob_thold in enumerate(prob_thold_list):
        y_pred, y_scores = dbclass.predict(ds_test['data'], prob_thold, return_labels=False)
        confusion_matrix = build_confusion_matrix(ds_test['target'], y_pred, labels=dbclass.pgc.keys())
        class_metrics = get_dbclass_metrics(confusion_matrix, ds_test['target'], y_pred, y_scores, prob_thold)

        accuracy_vec[t_index] = class_metrics['accuracy']
        prob_score_vec[t_index] = class_metrics['prob_score']
        confidence_vec[t_index] = class_metrics['confidence']
        rejection_vec[t_index] = class_metrics['rejection']

        if class_metrics['confidence'] > max_train_confidence:
            max_train_confidence = class_metrics['confidence']
            best_prob_thold = prob_thold
            best_metrics = class_metrics

        if not pca_test:
            if len(prob_thold_list) <= 11:
                print("\nProbability score threshold:", prob_thold)
                print('confidence:', class_metrics['confidence'])
                print(confusion_matrix)
            elif int(100*(t_index/len(prob_thold_list))) % 10 == 0:
                print(str(int(100*(t_index/len(prob_thold_list)))) + '% concluído...')

    if not pca_test:
        x_array = prob_thold_list
        Y_arrays = (accuracy_vec, prob_score_vec, confidence_vec, rejection_vec)
        datalabels = ["Accuracy", "Probability score", "Confidence", "Rejection"]
        fig_name = "Classifier trainning metrics"
        plot_train_metris(x_array, Y_arrays, datalabels, fig_name)

    return best_prob_thold, best_metrics


def build_confusion_matrix(y_true, y_pred, labels=[], normalize=True):
    """
    """

    if len(labels) > 0:
        n_classes = len(labels) + 1
    else:
        n_classes = len(np.unique(y_pred)) + 1

    confusion_matrix = np.zeros([n_classes, n_classes])
    for y, y_hat in zip(y_true, y_pred):
        confusion_matrix[y][y_hat] += 1

    if normalize is True:
        percent_convert = 1/len(y_true)
        confusion_matrix = np.round(confusion_matrix*(percent_convert), 2)

    return confusion_matrix


def get_dbclass_metrics(confusion_matrix, y_true, y_pred, y_scores, prob_thold):
    """
    """

    class_metrics = {}
    class_metrics['accuracy'] = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    non_negclass_scores = y_scores[np.where(y_pred != -1)]
    if len(non_negclass_scores) > 0:
        class_metrics['prob_score'] = np.sum(non_negclass_scores)/len(non_negclass_scores)
    else:
        class_metrics['prob_score'] = 1
    if len(np.where(y_true == -1)[0]) > 0 or len(np.where(y_pred == -1)[0]) > 0:
        class_metrics['rejection'] = np.sum(confusion_matrix[:, -1])/np.sum(confusion_matrix)
    else:
        class_metrics['rejection'] = 0

    # Classifier confidence level:
    avg_confidence = (class_metrics['accuracy'] + class_metrics['prob_score']) / 2
    weighted_confidence = prob_thold * class_metrics['accuracy'] + (1 - prob_thold) * class_metrics['prob_score']
    class_metrics['confidence'] = min([avg_confidence, weighted_confidence])

    for metric in class_metrics.keys():
        class_metrics[metric] = round(class_metrics[metric], 6)

    return class_metrics


def get_class_metrics(confusion_matrix):
    """
    """

    class_metrics = {}
    class_metrics['accuracy'] = np.trace(confusion_matrix)/np.sum(confusion_matrix)

    for metric in class_metrics.keys():
        class_metrics[metric] = round(class_metrics[metric], 6)

    return class_metrics


def plot_train_metris(x_array, Y_arrays, datalabels, fig_name):
    '''
    '''
    plt.close(fig_name)
    plt.figure(fig_name)
    for y_array, label in zip(Y_arrays, datalabels):
        if len(x_array) > 11:
            plt.plot(x_array, y_array, label=label, linewidth=2, zorder=5)
        else:
            plt.plot(x_array, y_array, label=label, linewidth=2, linestyle='--', marker='s', zorder=5)
    plt.xlabel('Probability threshold')
    plt.ylabel('Performance')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc='center left', ncol=1, prop={'size': 12})
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_pca_data(data, n_components, svd_solver='full', whiten=True):
    """
    """

    # Compute a PCA on the face dataset (treated as unlabeled dataset):
    # unsupervised feature extraction / dimensionality reduction
    pca = PCA(n_components=n_components, svd_solver=svd_solver,
              whiten=whiten).fit(data)
    # Projecting the input data on the eigenfaces orthonormal basis:
    pca_data = pca.transform(data)
    
    return pca_data


def perform_pca_test(ds, pca_limit=200):
    """
    """

    ds_pca = ds.copy()
    n_components_max = min([ds['data'].shape[1], pca_limit])
    pca_step = min([10, int(n_components_max/20)])
    n_components = pca_step
    n_components_best = 0
    max_confidence = 0
    
    while n_components < n_components_max:
        print('Actual n_components:', n_components, '- max:', n_components_max)
        ds_pca['data'] = get_pca_data(ds['data'], n_components=n_components)
        ds_train, ds_test = dtset.split_data(ds_pca, prop_train=0.8)
    
        # Trainning the model using cross validation in the trainning dataset:
        dbclass = DBCLASS()
        prob_thold_list = np.arange(0.45, 0.55, 0.01)
        best_prob_thold, class_metrics = cross_validation_trainning(dbclass, ds_train, ds_test, prob_thold_list, pca_test=True)
        if class_metrics['confidence'] > max_confidence:
            max_confidence = class_metrics['confidence']
            n_components_best = n_components

        n_components += pca_step

    return n_components_best, max_confidence


def dbclass_model_test(dbclass, ds, noclass_label="Unknown", cmap='Blues'):
    """
    """

    y_pred, y_scores = dbclass.predict(ds['data'], return_labels=False)
    confusion_matrix = build_confusion_matrix(ds['target'], y_pred, labels=dbclass.pgc.keys())
    class_metrics = get_dbclass_metrics(confusion_matrix, ds['target'], y_pred, y_scores, dbclass.prob_thold)

    ds_pred = print_classification_report(ds, y_pred, noclass_label)

    print('\nConfusion matrix:')
    vmin = min([0, confusion_matrix.min()])
    vmax = max([confusion_matrix.max()])    
    plt.figure()
    sns.heatmap(confusion_matrix, annot=np.round(confusion_matrix, 2), cmap=cmap, vmin=vmin, vmax=vmax)
    target_names = [str(x) for x in dbclass.target_names] + [dbclass.unk_class_label]
    target_idxs = [i+0.5 for i in range(len(target_names))]
    plt.xticks(target_idxs, target_names, rotation=-90)
    plt.yticks(target_idxs, target_names, rotation=0)
    plt.tight_layout()
    plt.show()


    print('\nIndex and Target label:')
    for index, tgt_name in enumerate(ds['target_names']):
        print('Index:', index,'Label:', tgt_name)
    print("\nProbability score threshold:", dbclass.prob_thold)
    for metric in class_metrics.keys():
        print(metric, class_metrics[metric])

    dtset.plot_dataset(ds_pred, feat_index=(0, 1), labeled=True, fig_name="model test")


def print_classification_report(ds, y_pred, noclass_label="Unknown"):
    """
    """

    ds_pred = ds.copy()
    if len(np.where(ds['target'] == -1)[0]) > 0 or len(np.where(y_pred == -1)[0]) > 0:
        report_y_true = np.where(ds['target'] == -1, len(ds['target_names']), ds['target'])
        report_y_pred = np.where(y_pred == -1, len(ds['target_names']), y_pred)
        report_target_names = np.append(ds['target_names'], noclass_label)
    else:
        report_y_true = ds['target']
        report_y_pred = y_pred
        report_target_names = ds['target_names']
    report_target_names = np.array([str(name) for name in report_target_names])
    ds_pred['target'] = y_pred
    ds_pred['target_names'] = report_target_names
    print(classification_report(report_y_true, report_y_pred, target_names=report_target_names))

    return ds_pred
