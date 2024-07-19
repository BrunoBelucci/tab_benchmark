import pytest
from tests.utils import generate_data_interesting_parameters, _test_fit_fn, _test_predict_fn, _test_predict_proba_fn
from tab_benchmark.models.lightgbm import LGBMClassifier, LGBMRegressor


classifier = LGBMClassifier
regressor = LGBMRegressor


@pytest.mark.parametrize("n_features, n_cat_features, samples, n_classes, max_cat_dim, task",
                         generate_data_interesting_parameters)
def test_fit_classifier(n_features, n_cat_features, samples, n_classes, max_cat_dim, task, tmp_path):
    if task == 'multi_regression':
        pytest.skip("LightGBM does not support multiregression")
    _test_fit_fn(classifier, {}, n_features, n_cat_features, samples, n_classes, max_cat_dim, task)


@pytest.mark.parametrize("n_features, n_cat_features, samples, n_classes, max_cat_dim, task",
                         generate_data_interesting_parameters)
def test_fit_regressor(n_features, n_cat_features, samples, n_classes, max_cat_dim, task, tmp_path):
    if task == 'multi_regression':
        pytest.skip("LightGBM does not support multiregression")
    _test_fit_fn(regressor, {}, n_features, n_cat_features, samples, n_classes, max_cat_dim, task)


@pytest.mark.parametrize("task", ['classification', 'binary_classification', 'regression', 'multi_regression'])
def test_predict_regressor(task, tmp_path):
    if task == 'multi_regression':
        pytest.skip("LightGBM does not support multiregression")
    _test_predict_fn(regressor, {}, task)


@pytest.mark.parametrize("task", ['classification', 'binary_classification', 'regression', 'multi_regression'])
def test_predict_classifier(task, tmp_path):
    if task == 'multi_regression':
        pytest.skip("LightGBM does not support multiregression")
    _test_predict_fn(classifier, {}, task)


@pytest.mark.parametrize("task", ['classification', 'binary_classification'])
def test_predict_proba_classifier(task, tmp_path):
    _test_predict_proba_fn(classifier, {}, task)

