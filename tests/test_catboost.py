import pytest
from tests.utils import generate_data_interesting_parameters, _test_fit_fn, _test_predict_fn, _test_predict_proba_fn
from tab_benchmark.models.catboost import TabBenchmarkCatBoostRegressor, TabBenchmarkCatBoostClassifier


classifier = TabBenchmarkCatBoostClassifier
regressor = TabBenchmarkCatBoostRegressor
kwargs = dict(iterations=100, allow_writing_files=False)


@pytest.mark.parametrize("n_features, n_cat_features, samples, n_classes, max_cat_dim, task",
                         generate_data_interesting_parameters)
def test_fit_classifier(n_features, n_cat_features, samples, n_classes, max_cat_dim, task, tmp_path):
    _test_fit_fn(classifier, kwargs, n_features, n_cat_features, samples, n_classes, max_cat_dim, task)


@pytest.mark.parametrize("n_features, n_cat_features, samples, n_classes, max_cat_dim, task",
                         generate_data_interesting_parameters)
def test_fit_regressor(n_features, n_cat_features, samples, n_classes, max_cat_dim, task, tmp_path):
    _test_fit_fn(regressor, kwargs, n_features, n_cat_features, samples, n_classes, max_cat_dim, task)


@pytest.mark.parametrize("task", ['classification', 'binary_classification', 'regression', 'multi_regression'])
def test_predict_regressor(task, tmp_path):
    _test_predict_fn(regressor, kwargs, task)


@pytest.mark.parametrize("task", ['classification', 'binary_classification', 'regression', 'multi_regression'])
def test_predict_classifier(task, tmp_path):
    _test_predict_fn(classifier, kwargs, task)


@pytest.mark.parametrize("task", ['classification', 'binary_classification'])
def test_predict_proba_classifier(task, tmp_path):
    _test_predict_proba_fn(classifier, kwargs, task)

