"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>

This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
import threading
import numpy as np
from numpy import ndarray
from joblib import effective_n_jobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.parallel import Parallel, delayed

def _combine_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    #prediction = predict(X, check_input=False)
    
    outcomes = []
    for p in predict(X, check_input=False):
        p_prime = np.zeros(p.shape[0])
        p_prime[np.argmax(p)] = 1
        outcomes.append(p_prime)
    
    with lock:
        if len(out) == 1:
            out[0] += outcomes
        else:
            for i in range(len(out)):
                p_prime = np.zeros(outcomes[i].shape[0])
                p_prime[np.argmax(outcomes[i])] = 1
                out[i] += p_prime[i]
                
def _partition(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs, dtype=int)
    n_estimators_per_job[: n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()

"""
    As reported in https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py#L890
    the RandomForestClassifier from sklearn module does not implement majority voting. Rather, The predicted class probabilities of an input
    sample are computed as the mean predicted class probabilities of the trees in the forest. The class probability of a single tree is the 
    fraction of samples of the same class in a leaf.

    This behavior differs from those from other ML frameworks, such as KNOME or Weka. The following class implements majority-voting based
    Random Forest Classifiers (RandomForestClassifierMV) on top of the RandomForestClassifier from sklearn module.
"""
class RandomForestClassifierMV(RandomForestClassifier):
    def __init__(self, n_estimators = 100, *, criterion = "gini", max_depth  = None, min_samples_split = 2, min_samples_leaf = 1, min_weight_fraction_leaf = 0, max_features = "sqrt", max_leaf_nodes = None, min_impurity_decrease = 0, bootstrap = True, oob_score = False, n_jobs = None, random_state = None, verbose = 0, warm_start = False, class_weight = None, ccp_alpha = 0, max_samples = None) -> None:
        super().__init__(n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
        
    def fake(self):    
        self.__class__ = RandomForestClassifier #! yeah, I know... I need this for PMML export! (Filippo dixit)
        
    def predict_proba(self, X) -> ndarray:
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of such arrays
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        
        check_is_fitted(self)
        X = self._validate_X_predict(X)
        n_jobs, _, _ = _partition(self.n_estimators, self.n_jobs)
        all_proba = [ np.zeros((X.shape[0], j), dtype=np.float64) for j in np.atleast_1d(self.n_classes_) ]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(delayed(_combine_prediction)(e.predict_proba, X, all_proba, lock) for e in self.estimators_)
        
        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba
    