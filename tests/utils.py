import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
import pytest

generate_data_interesting_parameters = [
    # "normal" cases
    (10, 5, 100, 5, 10, 'classification'),
    (10, 5, 100, 2, 10, 'binary_classification'),
    (10, 5, 100, 2, 10, 'multi_regression'),
    (10, 5, 100, 1, 10, 'regression'),
    # only categorical features
    (10, 10, 100, 5, 10, 'classification'),
    (10, 10, 100, 2, 10, 'binary_classification'),
    (10, 10, 100, 2, 10, 'multi_regression'),
    (10, 10, 100, 1, 10, 'regression'),
    # only numerical features
    (10, 0, 100, 5, 10, 'classification'),
    (10, 0, 100, 2, 10, 'binary_classification'),
    (10, 0, 100, 2, 10, 'multi_regression'),
    (10, 0, 100, 1, 10, 'regression'),

    # more features than samples
    # "normal" cases
    (100, 5, 50, 5, 10, 'classification'),
    (100, 5, 50, 2, 10, 'binary_classification'),
    (100, 5, 50, 2, 10, 'multi_regression'),
    (100, 5, 50, 1, 10, 'regression'),
    # only categorical features
    (100, 100, 50, 5, 10, 'classification'),
    (100, 100, 50, 2, 10, 'binary_classification'),
    (100, 100, 50, 2, 10, 'multi_regression'),
    (100, 100, 50, 1, 10, 'regression'),
    # only numerical features
    (100, 0, 50, 5, 10, 'classification'),
    (100, 0, 50, 2, 10, 'binary_classification'),
    (100, 0, 50, 2, 10, 'multi_regression'),
    (100, 0, 50, 1, 10, 'regression'),

    # more classes than samples

    # "normal" cases
    (10, 5, 50, 100, 10, 'classification'),
    (10, 5, 50, 100, 10, 'multi_regression'),
    # only categorical features
    (10, 10, 50, 100, 10, 'classification'),
    (10, 10, 50, 100, 10, 'multi_regression'),
    # only numerical features
    (10, 0, 50, 100, 10, 'classification'),
    (10, 0, 50, 100, 10, 'multi_regression'),

    # more categorical dimensions than samples

    # "normal" cases
    (10, 5, 50, 5, 100, 'classification'),
    (10, 5, 50, 2, 100, 'binary_classification'),
    (10, 5, 50, 2, 100, 'multi_regression'),
    (10, 5, 50, 1, 100, 'regression'),
    # only categorical features
    (10, 10, 50, 5, 100, 'classification'),
    (10, 10, 50, 2, 100, 'binary_classification'),
    (10, 10, 50, 2, 100, 'multi_regression'),
    (10, 10, 50, 1, 100, 'regression'),
    # only numerical features
    (10, 0, 50, 5, 100, 'classification'),
    (10, 0, 50, 2, 100, 'binary_classification'),
    (10, 0, 50, 2, 100, 'multi_regression'),
    (10, 0, 50, 1, 100, 'regression'),
]


def generate_data(n_features, n_cat_features, samples, n_classes, max_cat_dim, task):
    np.random.seed(42)
    cat_features_idx = np.random.choice(list(range(n_features)), size=n_cat_features, replace=False)
    cat_features_names = [str(i) for i in cat_features_idx]
    cont_features_names = [str(i) for i in range(n_features) if i not in cat_features_idx]
    orderly_features_names = [str(i) for i in range(n_features)]
    cat_dims = np.random.randint(1, max_cat_dim, n_cat_features)
    X = pd.DataFrame()
    i_cat_dims_it = 0
    for feature in range(n_features):
        if feature in cat_features_idx:
            X[str(feature)] = np.random.randint(0, cat_dims[i_cat_dims_it], samples)
            i_cat_dims_it += 1
        else:
            X[str(feature)] = np.random.rand(samples)
    if task in ('classification', 'binary_classification'):
        y = pd.DataFrame(np.random.randint(0, n_classes, samples), columns=['target'])
    else:
        y = pd.DataFrame(np.random.rand(samples, n_classes), columns=['target_' + str(i) for i in range(n_classes)])
    return X, y, cat_features_idx, cat_features_names, cont_features_names, orderly_features_names


def _test_fit_fn(model_class, model_kwargs, n_features, n_cat_features, samples, n_classes, max_cat_dim, task):
    X, y, cat_features_idx, cat_features_names, cont_features_names, orderly_features_names = generate_data(
        n_features, n_cat_features, samples, n_classes, max_cat_dim, task)
    fit_model(model_class, model_kwargs, X, y, cat_features_names, cont_features_names,
              orderly_features_names, task)


def fit_model(model_class, model_kwargs, X_train, y_train, cat_features_names, cont_features_names,
              orderly_features_names, task):
    if (model_class._estimator_type == 'classifier' and task in ('classification', 'binary_classification')) or (
            model_class._estimator_type == 'regressor' and task in ('regression', 'multi_regression')):
        if task == 'classification' and max(y_train.value_counts()) < 10:
            model_kwargs['categorical_target_min_frequency'] = 1
        model = model_class(**model_kwargs)
        model.create_preprocess_pipeline(task, cat_features_names, cont_features_names, orderly_features_names)
        model_pipeline = model.create_model_pipeline()
        model_pipeline.fit(X_train, y_train, target_preprocess_and_estimator__cat_features=cat_features_names,
                           target_preprocess_and_estimator__task=task)
        return model_pipeline
    else:
        pytest.skip('Model class and task do not match')


def _test_predict_fn(model_class, model_kwargs, task):
    if task == 'classification':
        n_classes = 5
    elif task == 'binary_classification':
        n_classes = 2
    elif task == 'regression':
        n_classes = 1
    elif task == 'multi_regression':
        n_classes = 3
    else:
        raise ValueError('Unknown task')
    X, y, cat_features_idx, cat_features_names, cont_features_names, orderly_features_names = generate_data(10, 5, 100,
                                                                                                            n_classes,
                                                                                                            10, task)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = fit_model(model_class, model_kwargs, X_train, y_train, cat_features_names, cont_features_names,
                      orderly_features_names, task)
    y_pred_1 = model.predict(X_test)
    y_pred_2 = model.predict(X_test)
    assert np.isclose(y_pred_1.astype(float), y_pred_2.astype(float)).all()


def _test_predict_proba_fn(model_class, model_kwargs, task):
    if task == 'classification':
        n_classes = 5
    elif task == 'binary_classification':
        n_classes = 2
    else:
        raise ValueError('Unknown task')
    X, y, cat_features_idx, cat_features_names, cont_features_names, orderly_features_names = generate_data(10, 5, 100,
                                                                                                            n_classes,
                                                                                                            10, task)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = fit_model(model_class, model_kwargs, X_train, y_train, cat_features_names, cont_features_names,
                      orderly_features_names, task)
    y_pred_1 = model.predict_proba(X_test)
    y_pred_2 = model.predict_proba(X_test)
    assert np.isclose(y_pred_1, y_pred_2).all()
