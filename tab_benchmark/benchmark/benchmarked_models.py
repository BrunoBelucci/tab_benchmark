from functools import partial
from tab_benchmark.dnns.architectures.utils import GLU, initialize_glu_, init_snn, GeneralReLU
from tab_benchmark.dnns.callbacks.one_cycle_lr import OneCycleLR, AutomaticOneCycleLR
from tab_benchmark.models.sk_learn_models import all_models as all_sklearn_models
from tab_benchmark.models.xgboost import XGBClassifier, XGBRegressor
from tab_benchmark.models.lightgbm import LGBMClassifier, LGBMRegressor
from tab_benchmark.models.catboost import CatBoostClassifier, CatBoostRegressor
from tab_benchmark.models.dnn_models import MLPModel, ResNetModel, TransformerModel
from torch import nn

models_dict = {model_cls.__name__: (model_cls, {}) for model_cls in all_sklearn_models}

n_estimators_gbdt = 10000
early_stopping_patience_gbdt = 100

XGB_params = dict(
    n_estimators=n_estimators_gbdt,
    auto_early_stopping=True,
    early_stopping_rounds=early_stopping_patience_gbdt,
    learning_rate=0.3,
    reg_lambda=1.0,
    reg_alpha=1e-10,
    gamma=1e-1,
    colsample_bylevel=1.0,
    colsample_bynode=1.0,
    colsample_bytree=1.0,
    max_depth=6,
    max_delta_step=0.0,
    min_child_weight=1.0,
    subsample=1.0,
    tree_method='auto'
)

LGBM_params = dict(
    n_estimators=n_estimators_gbdt,
    auto_early_stopping=True,
    early_stopping_rounds=early_stopping_patience_gbdt,
    learning_rate=0.1,
    reg_lambda=0.0,
    reg_alpha=0.0,
    min_split_gain=0.0,
    colsample_bytree=1.0,
    feature_fraction_bynode=1.0,
    max_depth=-1,
    num_leaves=31,
    min_child_samples=20,
    max_delta_step=0.0,
    min_child_weight=1e-3,
    subsample=1.0,
    subsample_freq=0,
)

CatBoost_params = dict(
    iterations=n_estimators_gbdt,
    auto_early_stopping=True,
    early_stopping_rounds=early_stopping_patience_gbdt,
    random_strength=1.0,
    l2_leaf_reg=3.0,
    bagging_temperature=1.0,
)


models_dict.update({
    XGBClassifier.__name__: (XGBClassifier, XGB_params),
    XGBRegressor.__name__: (XGBRegressor, XGB_params),
    LGBMClassifier.__name__: (LGBMClassifier, LGBM_params),
    LGBMRegressor.__name__: (LGBMRegressor, LGBM_params),
    CatBoostClassifier.__name__: (CatBoostClassifier, CatBoost_params),
    CatBoostRegressor.__name__: (CatBoostRegressor, CatBoost_params),
})


def init_GLU_kwargs(model):
    # lazy initialization
    if issubclass(model, MLPModel):
        return dict(activation_fn=GLU(256), initialization_fn=partial(initialize_glu_, input_dim=256, output_dim=256))
    elif issubclass(model, ResNetModel):
        return dict(activation_fn_1=GLU(256), activation_fn_2=GLU(256),
                    initialization_fn_1=partial(initialize_glu_, input_dim=256, output_dim=256),
                    initialization_fn_2=partial(initialize_glu_, input_dim=256, output_dim=256))


models_dict.update(
    {
        'MLPModel': (MLPModel, {}),
        'ResNetModel': (ResNetModel, {}),
        'TransformerModel': (TransformerModel, {}),
        'MLP_Deeper': (MLPModel, {'n_layers': 8}),
        'MLP_Wider': (MLPModel, {'hidden_dim': 512}),
        'ResNet_Deeper': (ResNetModel, {'n_blocks': 4}),
        'ResNet_Wider': (ResNetModel, {'blocks_dim': 512}),
        'MLP_GLU': (MLPModel, init_GLU_kwargs),
        'MLP_SNN': (MLPModel, dict(activation_fn=nn.SELU(), initialization_fn=init_snn,
                                   dropout_module_class=nn.AlphaDropout, norm_module_class=nn.Identity)),
        'ResNet_SNN': (ResNetModel, dict(activation_fn_1=nn.SELU(), activation_fn_2=nn.SELU(),
                                         initialization_fn_1=init_snn, initialization_fn_2=init_snn,
                                         dropout_module_class_1=nn.AlphaDropout,
                                         dropout_module_class_2=nn.AlphaDropout,
                                         norm_module_class_1=nn.Identity, norm_module_class_2=nn.Identity)),
        'ResNet_GLU': (ResNetModel, init_GLU_kwargs),
        'MLP_GReLUOneCycleLR': (MLPModel, dict(
            activation_fn=GeneralReLU(0.1, 0.4), initialization_fn=partial(nn.init.kaiming_normal_, a=0.1),
            lit_callbacks_tuples=[(OneCycleLR, dict(max_lr=1e-3)), ],
            early_stopping_patience=0, n_iter=100, lit_trainer_kwargs=dict(max_epochs=100))),
        'ResNet_GReLUOneCycleLR': (ResNetModel, dict(
            activation_fn_1=GeneralReLU(0.1, 0.4), activation_fn_2=GeneralReLU(0.1, 0.4),
            initialization_fn_1=partial(nn.init.kaiming_normal_, a=0.1),
            initialization_fn_2=partial(nn.init.kaiming_normal_, a=0.1),
            lit_callbacks_tuples=[(OneCycleLR, dict(max_lr=1e-3)), ],
            early_stopping_patience=0, n_iter=100, lit_trainer_kwargs=dict(max_epochs=100))),
        'MLP_GReLUAutoOneCycleLR': (MLPModel, dict(
            activation_fn=GeneralReLU(0.1, 0.4), initialization_fn=partial(nn.init.kaiming_normal_, a=0.1),
            lit_callbacks_tuples=[(AutomaticOneCycleLR, dict(suggestion_method='steep',
                                                             early_stop_threshold=None)), ],
            early_stopping_patience=0, n_iter=100, lit_trainer_kwargs=dict(max_epochs=100))),
        'ResNet_GReLUAutoOneCycleLR': (ResNetModel, dict(
            activation_fn_1=GeneralReLU(0.1, 0.4), activation_fn_2=GeneralReLU(0.1, 0.4),
            initialization_fn_1=partial(nn.init.kaiming_normal_, a=0.1),
            initialization_fn_2=partial(nn.init.kaiming_normal_, a=0.1),
            lit_callbacks_tuples=[(AutomaticOneCycleLR, dict(suggestion_method='steep',
                                                             early_stop_threshold=None)), ],
            early_stopping_patience=0, n_iter=100, lit_trainer_kwargs=dict(max_epochs=100))),
    }
)
