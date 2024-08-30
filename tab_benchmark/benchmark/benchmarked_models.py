from functools import partial
from tab_benchmark.dnns.architectures.utils import GLU, initialize_glu_, init_snn, GeneralReLU
from tab_benchmark.dnns.callbacks.one_cycle_lr import OneCycleLR, AutomaticOneCycleLR
from tab_benchmark.models.sk_learn_models import all_models as all_sklearn_models
from tab_benchmark.models.xgboost import TabBenchmarkXGBClassifier, TabBenchmarkXGBRegressor
from tab_benchmark.models.lightgbm import TabBenchmarkLGBMClassifier, TabBenchmarkLGBMRegressor
from tab_benchmark.models.catboost import TabBenchmarkCatBoostClassifier, TabBenchmarkCatBoostRegressor
from tab_benchmark.models.dnn_models import TabBenchmarkMLP, TabBenchmarkResNet, TabBenchmarkTransformer
from torch import nn

models_dict = {model_cls.__name__: (model_cls, {}) for model_cls in all_sklearn_models}


models_dict.update({
    TabBenchmarkXGBClassifier.__name__: (TabBenchmarkXGBClassifier, TabBenchmarkXGBClassifier.get_recommended_params()),
    TabBenchmarkXGBRegressor.__name__: (TabBenchmarkXGBRegressor, TabBenchmarkXGBRegressor.get_recommended_params()),
    TabBenchmarkLGBMClassifier.__name__: (TabBenchmarkLGBMClassifier, TabBenchmarkLGBMClassifier.get_recommended_params()),
    TabBenchmarkLGBMRegressor.__name__: (TabBenchmarkLGBMRegressor, TabBenchmarkLGBMRegressor.get_recommended_params()),
    TabBenchmarkCatBoostClassifier.__name__: (TabBenchmarkCatBoostClassifier,
                                              TabBenchmarkCatBoostClassifier.get_recommended_params()),
    TabBenchmarkCatBoostRegressor.__name__: (TabBenchmarkCatBoostRegressor,
                                             TabBenchmarkCatBoostRegressor.get_recommended_params()),
})


def init_GLU_kwargs(model):
    # lazy initialization
    if issubclass(model, TabBenchmarkMLP):
        return dict(activation_fn=GLU(256), initialization_fn=partial(initialize_glu_, input_dim=256, output_dim=256))
    elif issubclass(model, TabBenchmarkResNet):
        return dict(activation_fn_1=GLU(256), activation_fn_2=GLU(256),
                    initialization_fn_1=partial(initialize_glu_, input_dim=256, output_dim=256),
                    initialization_fn_2=partial(initialize_glu_, input_dim=256, output_dim=256))


models_dict.update(
    {
        'TabBenchmarkMLP': (TabBenchmarkMLP, TabBenchmarkMLP.get_recommended_params()),
        'TabBenchmarkResNet': (TabBenchmarkResNet, {}),
        'TabBenchmarkTransformer': (TabBenchmarkTransformer, {}),
        'TabBenchmarkMLP_Deeper': (TabBenchmarkMLP, {'n_layers': 8}),
        'TabBenchmarkMLP_Wider': (TabBenchmarkMLP, {'hidden_dim': 512}),
        'TabBenchmarkResNet_Deeper': (TabBenchmarkResNet, {'n_blocks': 4}),
        'TabBenchmarkResNet_Wider': (TabBenchmarkResNet, {'blocks_dim': 512}),
        'TabBenchmarkMLP_GLU': (TabBenchmarkMLP, init_GLU_kwargs),
        'TabBenchmarkMLP_SNN': (TabBenchmarkMLP, dict(activation_fn=nn.SELU(), initialization_fn=init_snn,
                                   dropout_module_class=nn.AlphaDropout, norm_module_class=nn.Identity)),
        'TabBenchmarkResNet_SNN': (TabBenchmarkResNet, dict(activation_fn_1=nn.SELU(), activation_fn_2=nn.SELU(),
                                         initialization_fn_1=init_snn, initialization_fn_2=init_snn,
                                         dropout_module_class_1=nn.AlphaDropout,
                                         dropout_module_class_2=nn.AlphaDropout,
                                         norm_module_class_1=nn.Identity, norm_module_class_2=nn.Identity)),
        'TabBenchmarkResNet_GLU': (TabBenchmarkResNet, init_GLU_kwargs),
        'TabBenchmarkMLP_GReLUOneCycleLR': (TabBenchmarkMLP, dict(
            activation_fn=GeneralReLU(0.1, 0.4), initialization_fn=partial(nn.init.kaiming_normal_, a=0.1),
            lit_callbacks_tuples=[(OneCycleLR, dict(max_lr=1e-3)), ],
            early_stopping_patience=0, n_iter=100, lit_trainer_params=dict(max_epochs=100))),
        'TabBenchmarkResNet_GReLUOneCycleLR': (TabBenchmarkResNet, dict(
            activation_fn_1=GeneralReLU(0.1, 0.4), activation_fn_2=GeneralReLU(0.1, 0.4),
            initialization_fn_1=partial(nn.init.kaiming_normal_, a=0.1),
            initialization_fn_2=partial(nn.init.kaiming_normal_, a=0.1),
            lit_callbacks_tuples=[(OneCycleLR, dict(max_lr=1e-3)), ],
            early_stopping_patience=0, n_iter=100, lit_trainer_params=dict(max_epochs=100))),
        'TabBenchmarkMLP_GReLUAutoOneCycleLR': (TabBenchmarkMLP, dict(
            activation_fn=GeneralReLU(0.1, 0.4), initialization_fn=partial(nn.init.kaiming_normal_, a=0.1),
            lit_callbacks_tuples=[(AutomaticOneCycleLR, dict(suggestion_method='steep',
                                                             early_stop_threshold=None)), ],
            early_stopping_patience=0, n_iter=100, lit_trainer_params=dict(max_epochs=100))),
        'TabBenchmarkResNet_GReLUAutoOneCycleLR': (TabBenchmarkResNet, dict(
            activation_fn_1=GeneralReLU(0.1, 0.4), activation_fn_2=GeneralReLU(0.1, 0.4),
            initialization_fn_1=partial(nn.init.kaiming_normal_, a=0.1),
            initialization_fn_2=partial(nn.init.kaiming_normal_, a=0.1),
            lit_callbacks_tuples=[(AutomaticOneCycleLR, dict(suggestion_method='steep',
                                                             early_stop_threshold=None)), ],
            early_stopping_patience=0, n_iter=100, lit_trainer_params=dict(max_epochs=100))),
    }
)
