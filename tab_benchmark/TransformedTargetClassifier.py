# Everything is copied from pycaret.internal.preprocess.target.TransformedTargetClassifier.py,
# pycaret.internal.preprocess.target.utils.py and pycaret.utils.generic.py

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import (
    _deprecate_positional_args,
    check_is_fitted,
    check_X_y,
)


def get_all_object_vars_and_properties(object):
    """
    Gets all class, static and dynamic attributes from an object.

    Calling ``vars()`` would only return static attributes.

    https://stackoverflow.com/a/59769926

    # TODO: Do both:
    # Option 1: Set fh before calling any model
    "C:\ProgramData\Anaconda3\envs\pycaret_dev\lib\site-packages\sktime\forecasting\base\_sktime.py", line 187
    def _set_fh(self, fh):
    # Option 2: Ignore the exceptions
    """
    d = {}
    for k in object.__dir__():
        try:
            if k[:2] != "__" and type(getattr(object, k, "")).__name__ != "method":
                d[k] = getattr(object, k, "")
        except Exception:
            pass
    return d


def is_fit_var(key):
    return key and (
        (key.endswith("_") and not key.startswith("_")) or (key in ["n_clusters"])
    )


class TargetTransformerMixin:
    @property
    def estimator(self):
        if hasattr(self, "regressor_"):
            return self.regressor_
        if hasattr(self, "regressor"):
            return self.regressor
        if hasattr(self, "classifier_"):
            return self.classifier_
        if hasattr(self, "classifier"):
            return self.classifier
        return self

    def _carry_over_estimator_fit_vars(self, fitted_estimator, ignore: list = None):
        if not ignore:
            ignore = set()
        else:
            ignore = set(ignore)
        if not hasattr(self, "_fit_vars"):
            self._fit_vars = set()
        self._clear_estimator_fit_vars(fitted_estimator)
        for k, v in get_all_object_vars_and_properties(fitted_estimator).items():
            if is_fit_var(k) and k not in ignore:
                try:
                    setattr(self, k, v)
                    self._fit_vars.add(k)
                except Exception:
                    pass

    def _clear_estimator_fit_vars(self, fitted_estimator, all: bool = False):
        if not hasattr(self, "_fit_vars"):
            self._fit_vars = set()
        vars_to_remove = []
        try:
            for var in self._fit_vars:
                if all or var not in get_all_object_vars_and_properties(
                    fitted_estimator
                ):
                    vars_to_remove.append(var)
            for var in vars_to_remove:
                try:
                    delattr(self, var)
                    self._fit_vars.remove(var)
                except Exception:
                    pass
        except Exception:
            pass


def get_estimator_from_meta_estimator(estimator):
    """
    If ``estimator`` is a meta estimator, get estimator inside.
    Otherwise return ``estimator``. Will try to return the fitted
    estimator first.
    """
    try:
        return estimator.estimator
    except Exception:
        return estimator


class TransformedTargetClassifier(
    TargetTransformerMixin, ClassifierMixin, BaseEstimator
):
    """Meta-estimator to classify on a transformed target.

    Parameters
    ----------
    classifier : object, default=None
        Classifier object such as derived from ``ClassifierMixin``. This
        classifier will automatically be cloned each time prior to fitting.

    transformer : object, default=None
        Estimator object such as derived from ``TransformerMixin``. Note that the
        transformer will be cloned during fitting. Also, the transformer is
        restricting ``y`` to be a numpy array.

    check_inverse : bool, default=True
        Whether to check that ``transform`` followed by ``inverse_transform``
        leads to the original targets.

    Attributes
    ----------
    classifier_ : object
        Fitted classifier.

    transformer_ : object
        Transformer used in ``fit`` and ``predict``.

    Notes
    -----
    This is a modification from TransformedTargetRegressor.

    See :ref:`sklearn.compose.TransformedTargetRegressor

    """

    @_deprecate_positional_args
    def __init__(self, classifier=None, *, transformer=None, check_inverse=True):
        self.classifier = classifier
        self.transformer = transformer
        self.check_inverse = check_inverse

    def _fit_transformer(self, y):
        """Check transformer and fit transformer.

        Create the default transformer, fit it and make additional inverse
        check on a subset (optional).

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        """
        if self.transformer is not None:
            self.transformer_ = clone(self.transformer)
        else:  # Just return the identity transformer
            self.transformer_ = LabelEncoder()

        self.transformer_.fit(y)
        if self.check_inverse:
            idx_selected = slice(None, None, max(1, y.shape[0] // 10))
            y_sel = _safe_indexing(y, idx_selected)
            y_sel_t = self.transformer_.transform(y_sel)

            if (np.ravel(y_sel) != self.transformer_.inverse_transform(y_sel_t)).any():
                warnings.warn(
                    "The provided functions or transformer are"
                    " not strictly inverse of each other. If"
                    " you are sure you want to proceed regardless"
                    ", set 'check_inverse=False'",
                    UserWarning,
                )

    def fit(self, X, y, **fit_params):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        **fit_params : dict
            Parameters passed to the ``fit`` method of the underlying
            classifier.


        Returns
        -------
        self : object
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values

        # y = check_array(y, accept_sparse=False, force_all_finite=True,
        #                 ensure_2d=False, dtype=None)

        # Store the classes seen during fit
        self.classes_ = np.unique(y)

        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._training_dim = y.ndim

        # transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self._fit_transformer(y_2d)

        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y_2d)
        # FIXME: a FunctionTransformer can return a 1D array even when validate
        # is set to True. Therefore, we need to check the number of dimension
        # first.
        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)

        if self.classifier is None:
            from sklearn.linear_model import LogisticRegression

            self.classifier_ = LogisticRegression()
        else:
            self.classifier_ = clone(self.classifier)

        self.classifier_.fit(X, y_trans, **fit_params)

        self._carry_over_estimator_fit_vars(
            self.classifier_, ignore=["classes_", "transformer_", "classifier_"]
        )

        return self

    def predict(self, X):
        """Predict using the base classifier, applying inverse.

        The classifier is used to predict and the
        ``inverse_transform`` is applied before returning the prediction.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_hat : ndarray of shape (n_samples,)
            Predicted values.

        """
        check_is_fitted(self)
        pred = self.classifier_.predict(X)
        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(pred.reshape(-1, 1))
        else:
            pred_trans = self.transformer_.inverse_transform(pred)
        if (
            self._training_dim == 1
            and pred_trans.ndim == 2
            and pred_trans.shape[1] == 1
        ):
            pred_trans = pred_trans.squeeze(axis=1)

        return pred_trans

    def _more_tags(self):
        return {"poor_score": True, "no_validation": True}

    @property
    def n_features_in_(self):
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() returns False the estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(
                    self.__class__.__name__
                )
            ) from nfe

        return self.classifier_.n_features_in_
