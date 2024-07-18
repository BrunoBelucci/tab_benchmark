from xgboost import XGBClassifier as OriginalXGBClassifier, XGBRegressor as OriginalXGBRegressor, XGBModel
from catboost import CatBoost
from lightgbm import LGBMModel
from tab_benchmark.models.factories import SimpleSkLearnFactory
from tab_benchmark.utils import (extends, train_test_split_forced, sequence_to_list, check_if_arg_in_args_of_fn,
                                 check_if_arg_in_kwargs_of_fn)


def fn_to_run_before_fit_for_gbdt(self, X, y, task, cat_features, *args, **kwargs):
    if self.auto_early_stopping:
        if self.task_ == 'classification' or self.task_ == 'binary_classification':
            stratify = y
        else:
            stratify = None
        X, X_valid, y, y_valid = train_test_split_forced(
            X, y,
            test_size_pct=self.early_stopping_validation_size,
            # random_state=self.random_seed,  this will be ensured by set_seeds
            stratify=stratify
        )
        eval_set = kwargs.get('eval_set', [])
        eval_set = sequence_to_list(eval_set)
        eval_set.append((X_valid, y_valid))
        kwargs['eval_set'] = eval_set
    if isinstance(self, XGBModel):
        if cat_features is not None:
            self.enable_categorical = True  # Categorical type must be set in dataframe
    elif isinstance(self, CatBoost):
        self.cat_features = cat_features
        self._init_params['cat_features'] = cat_features
    elif isinstance(self, LGBMModel):
        if cat_features is not None:
            if not check_if_arg_in_kwargs_of_fn('categorical_feature', **kwargs):
                i_possible_args = check_if_arg_in_args_of_fn(self.fit, 'categorical_feature', *args)
                if i_possible_args == False:
                    # categorical_feature is not in kwargs and not in args
                    kwargs['categorical_feature'] = cat_features
    return X, y, task, cat_features, args, kwargs


# Just to get the parameters of the XGBModel, because XGBClassifier and XGBRegressor do not show them
# This will initialize exactly the same way as XGBClassifier and XGBRegressor.
class XGBClassifier(OriginalXGBClassifier):
    __doc__ = OriginalXGBClassifier.__doc__

    @extends(XGBModel.__init__, map_default_values_change={'objective': 'binary:logistic'})
    def __init__(self, *args, **kwargs):
        XGBModel.__init__(self, *args, **kwargs)


class XGBRegressor(OriginalXGBRegressor):
    __doc__ = OriginalXGBRegressor.__doc__

    @extends(XGBModel.__init__, map_default_values_change={'objective': 'reg:squarederror'})
    def __init__(self, *args, **kwargs):
        XGBModel.__init__(self, *args, **kwargs)


# Now we wrap them with our API
XGBClassifier = SimpleSkLearnFactory.from_sk_cls(
    XGBClassifier,
    map_task_to_default_values={
        'classification': {'objective': 'multi:softmax', 'eval_metric': 'mlogloss'},
        'binary_classification': {'objective': 'binary:logistic', 'eval_metric': 'logloss'},
    },
    has_auto_early_stopping=True,
    fn_to_run_before_fit=fn_to_run_before_fit_for_gbdt,
    extended_init_kwargs={
        'categorical_encoder': None,
        'categorical_type': 'category',
        'data_scaler': None,
        'continuous_target_scaler': None,
    }
)

XGBRegressor = SimpleSkLearnFactory.from_sk_cls(
    XGBRegressor,
    map_default_values_change={
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    },
    has_auto_early_stopping=True,
    fn_to_run_before_fit=fn_to_run_before_fit_for_gbdt,
    extended_init_kwargs={
        'categorical_encoder': None,
        'categorical_type': 'category',
        'data_scaler': None,
        'continuous_target_scaler': None,
    }
)
