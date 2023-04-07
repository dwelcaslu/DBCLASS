# Third-party libraries:
import numpy as np
# My libraries:


class PGC():
    """
    PGC: Probabilistic Gaussian Classifier.
    """

    def __init__(self, pgc=None, prob_thold=0.5):
        """
        """
        self.pgc = pgc
        self.prob_thold = prob_thold

    def fit(self, ds):
        """
        This definition models the classes available in the dataset to
        gaussians , by feature, in order to represent each feature separately.
        """
        
        if 'feature_names' in ds:
            self.feature_names = ds['feature_names']
        else:
            self.feature_names = np.array(["$x_{}$".format(str(t + 1)) for t in np.unique(ds['target'])])

        self.target_names = ds['target_names']
        
        # Fitting the PGC to the data:
        pgc = {}
        data = self.split_classes(ds)
        for class_target in data.keys():
            pgc[class_target] = {}
            pgc[class_target]['mu'] = np.mean(data[class_target], axis=0)
            pgc[class_target]['sigma'] = np.std(data[class_target], axis=0)
        self.pgc = pgc

    def fitxy(self, X_train, y_train):
        """
        This definition models the classes available in the dataset to
        gaussians , by feature, in order to represent each feature separately.
        """
        
        # Fitting the PGC to the data:
        pgc = {}
        data = self.split_classes({"data": X_train, "target": y_train})
        for class_target in data.keys():
            pgc[class_target] = {}
            pgc[class_target]['mu'] = np.mean(data[class_target], axis=0)
            pgc[class_target]['sigma'] = np.std(data[class_target], axis=0)
        self.pgc = pgc

    def split_classes(self, ds):
        """
        This definition splits the data set between the classes.
        """
    
        data = {}
        for target in np.unique(ds['target']):
            data[target] = ds['data'][np.where(ds['target'] == target)]
    
        return data

    def predict(self, X, prob_thold=None, return_labels=True, noclass_label="Unknown"):
        """
        Given a set of data points, this definition attemps to predict the most
        appropriate label.
        """

        if prob_thold is None:
            prob_thold = self.prob_thold

        n_samples = X.shape[0]
        y_scores = np.zeros(n_samples)
        y_prediction_targets = -np.ones(n_samples, dtype=int)
        y_prediction_labels = np.array([noclass_label for x in range(n_samples)])

        for n, reg in enumerate(X):
            for class_target in self.pgc.keys():
                scores_array = np.zeros(len(reg))
                for index, val in enumerate(reg):
                    mu = self.pgc[class_target]['mu'][index]
                    sigma = self.pgc[class_target]['sigma'][index]
                    scores_array[index] = self.get_prob_score(val, mu, sigma)

                scores_array = scores_array[np.logical_not(np.isnan(scores_array))]
                class_score = np.mean(scores_array)

                if class_score > prob_thold and class_score > y_scores[n]:
                    y_scores[n] = class_score
                    y_prediction_targets[n] = class_target
                    y_prediction_labels[n] = self.target_names[class_target]

        if return_labels is True:
            y_predictions = y_prediction_labels
        else:
            y_predictions = y_prediction_targets

        return y_predictions, y_scores

    def get_prob_score(self, x, mu, sigma):
        """
        This difinition calculates the feature probability score.
        """
        if sigma > 0:
            x_score = self.get_gauss_dist_prob(x, mu, sigma)
            mu_score = self.get_gauss_dist_prob(mu, mu, sigma)
            feat_score = x_score/mu_score
        else:
            feat_score = np.nan
        return feat_score

    def get_gauss_dist_prob(self, x, mu, sigma):
        """
        Calculates the probability density in a gaussian distribution.
        """
        prob_x = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))
        return prob_x
