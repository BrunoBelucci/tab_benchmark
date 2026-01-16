from functools import partial
from typing import TypedDict, Any, Callable
from tab_benchmark.dnns.architectures.utils import GLU, initialize_glu_, init_snn, GeneralReLU
from tab_benchmark.dnns.callbacks.one_cycle_lr import OneCycleLR, AutomaticOneCycleLR
from tab_benchmark.models.sk_learn_models import all_models as all_sklearn_models
from tab_benchmark.models.xgboost import TabBenchmarkXGBClassifier, TabBenchmarkXGBRegressor
from tab_benchmark.models.lightgbm import TabBenchmarkLGBMClassifier, TabBenchmarkLGBMRegressor
from tab_benchmark.models.catboost import TabBenchmarkCatBoostClassifier, TabBenchmarkCatBoostRegressor
from tab_benchmark.models.dnn_models import (TabBenchmarkMLP, TabBenchmarkResNet, TabBenchmarkTransformer,
                                             TabBenchmarkSaint, TabBenchmarkNode, TabBenchmarkTabNet,
                                             TabBenchmarkTabTransformer)
from torch import nn
from copy import deepcopy


class ModelConfig(TypedDict):
    model_class: type
    model_params: dict[str, Any] | Callable
    search_space: dict[str, Any]
    default_values: list[Any]


models_dict: dict[str, ModelConfig] 

models_dict = {
    model_cls.__name__: ModelConfig(model_class=model_cls, model_params={}, search_space={}, default_values=[])
    for model_cls in all_sklearn_models
}

# enable probability in TabBenchmarkNuSVC
models_dict["TabBenchmarkNuSVC"] = ModelConfig(
    model_class=models_dict["TabBenchmarkNuSVC"]["model_class"],
    model_params={"probability": True},
    search_space=models_dict["TabBenchmarkNuSVC"]["search_space"],
    default_values=models_dict["TabBenchmarkNuSVC"]["default_values"],
)

models_gbdt = {
    TabBenchmarkXGBClassifier.__name__: ModelConfig(
        model_class=TabBenchmarkXGBClassifier,
        model_params=TabBenchmarkXGBClassifier.get_recommended_params(),
        search_space=TabBenchmarkXGBClassifier.create_search_space()[0],
        default_values=[TabBenchmarkXGBClassifier.create_search_space()[1]],
    ),
    TabBenchmarkXGBRegressor.__name__: ModelConfig(
        model_class=TabBenchmarkXGBRegressor,
        model_params=TabBenchmarkXGBRegressor.get_recommended_params(),
        search_space=TabBenchmarkXGBRegressor.create_search_space()[0],
        default_values=[TabBenchmarkXGBRegressor.create_search_space()[1]],
    ),
    TabBenchmarkLGBMClassifier.__name__: ModelConfig(
        model_class=TabBenchmarkLGBMClassifier,
        model_params=TabBenchmarkLGBMClassifier.get_recommended_params(),
        search_space=TabBenchmarkLGBMClassifier.create_search_space()[0],
        default_values=[TabBenchmarkLGBMClassifier.create_search_space()[1]],
    ),
    TabBenchmarkLGBMRegressor.__name__: ModelConfig(
        model_class=TabBenchmarkLGBMRegressor,
        model_params=TabBenchmarkLGBMRegressor.get_recommended_params(),
        search_space=TabBenchmarkLGBMRegressor.create_search_space()[0],
        default_values=[TabBenchmarkLGBMRegressor.create_search_space()[1]],
    ),
    TabBenchmarkCatBoostClassifier.__name__: ModelConfig(
        model_class=TabBenchmarkCatBoostClassifier,
        model_params=TabBenchmarkCatBoostClassifier.get_recommended_params(),
        search_space=TabBenchmarkCatBoostClassifier.create_search_space()[0],
        default_values=[TabBenchmarkCatBoostClassifier.create_search_space()[1]],
    ),
    TabBenchmarkCatBoostRegressor.__name__: ModelConfig(
        model_class=TabBenchmarkCatBoostRegressor,
        model_params=TabBenchmarkCatBoostRegressor.get_recommended_params(),
        search_space=TabBenchmarkCatBoostRegressor.create_search_space()[0],
        default_values=[TabBenchmarkCatBoostRegressor.create_search_space()[1]],
    ),
}

models_dict.update(models_gbdt)


MLP_recommended_params = TabBenchmarkMLP.get_recommended_params()
ResNet_recommended_params = TabBenchmarkResNet.get_recommended_params()
Transformer_recommended_params = TabBenchmarkTransformer.get_recommended_params()
Saint_recommended_params = TabBenchmarkSaint.get_recommended_params()
Node_recommended_params = TabBenchmarkNode.get_recommended_params()
TabNet_recommended_params = TabBenchmarkTabNet.get_recommended_params()
TabTransformer_recommended_params = TabBenchmarkTabTransformer.get_recommended_params()
MLP_params = deepcopy(MLP_recommended_params)
ResNet_params = deepcopy(ResNet_recommended_params)
Transformer_params = deepcopy(Transformer_recommended_params)
Saint_params = deepcopy(Saint_recommended_params)
Node_params = deepcopy(Node_recommended_params)
TabNet_params = deepcopy(TabNet_recommended_params)


def init_GLU_kwargs(model_cls, **kwargs):
    # lazy initialization
    initialization_fn = partial(initialize_glu_, input_dim=256, output_dim=256)
    setattr(initialization_fn, '__name__', 'initialize_glu_')
    if model_cls.__name__ == TabBenchmarkMLP.__name__:
        glu_kwargs = {**MLP_params.copy(),
                      **dict(activation_fns=GLU(256),
                             initialization_fns=initialization_fn)}
    elif model_cls.__name__ == TabBenchmarkResNet.__name__:
        glu_kwargs = {**ResNet_params.copy(),
                      **dict(activation_fns_1=GLU(256), activation_fns_2=GLU(256),
                             initialization_fns_1=initialization_fn,
                             initialization_fns_2=initialization_fn)}
    else:
        raise ValueError(f'Unknown model_cls: {model_cls}')
    glu_kwargs.update(kwargs)
    return glu_kwargs

models_dnns = {
    "TabBenchmarkMLP": ModelConfig(
        model_class=TabBenchmarkMLP,
        model_params=deepcopy(MLP_params),
        search_space=TabBenchmarkMLP.create_search_space()[0],
        default_values=[TabBenchmarkMLP.create_search_space()[1]],
    ),
    "TabBenchmarkResNet": ModelConfig(
        model_class=TabBenchmarkResNet,
        model_params=deepcopy(ResNet_params),
        search_space=TabBenchmarkResNet.create_search_space()[0],
        default_values=[TabBenchmarkResNet.create_search_space()[1]],
    ),
    "TabBenchmarkTransformer": ModelConfig(
        model_class=TabBenchmarkTransformer,
        model_params=deepcopy(Transformer_params),
        search_space=TabBenchmarkTransformer.create_search_space()[0],
        default_values=[TabBenchmarkTransformer.create_search_space()[1]],
    ),
    "TabBenchmarkSaint": ModelConfig(
        model_class=TabBenchmarkSaint,
        model_params=deepcopy(Saint_params),
        search_space=TabBenchmarkSaint.create_search_space()[0],
        default_values=[TabBenchmarkSaint.create_search_space()[1]],
    ),
    "TabBenchmarkNode": ModelConfig(
        model_class=TabBenchmarkNode,
        model_params=deepcopy(Node_params),
        search_space=TabBenchmarkNode.create_search_space()[0],
        default_values=[TabBenchmarkNode.create_search_space()[1]],
    ),
    "TabBenchmarkTabNet": ModelConfig(
        model_class=TabBenchmarkTabNet,
        model_params=deepcopy(TabNet_params),
        search_space=TabBenchmarkTabNet.create_search_space()[0],
        default_values=[TabBenchmarkTabNet.create_search_space()[1]],
    ),
    "TabBenchmarkTabTransformer": ModelConfig(
        model_class=TabBenchmarkTabTransformer,
        model_params=deepcopy(TabTransformer_recommended_params),
        search_space=TabBenchmarkTabTransformer.create_search_space()[0],
        default_values=[TabBenchmarkTabTransformer.create_search_space()[1]],
    ),
    "TabBenchmarkMLP_Deeper": ModelConfig(
        model_class=TabBenchmarkMLP,
        model_params={**deepcopy(MLP_params), **{"n_layers": 8}},
        search_space=TabBenchmarkMLP.create_search_space()[0],
        default_values=[TabBenchmarkMLP.create_search_space()[1]],
    ),
    "TabBenchmarkMLP_Wider": ModelConfig(
        model_class=TabBenchmarkMLP,
        model_params={**deepcopy(MLP_params), **{"hidden_dims": 512}},
        search_space=TabBenchmarkMLP.create_search_space()[0],
        default_values=[TabBenchmarkMLP.create_search_space()[1]],
    ),
    "TabBenchmarkResNet_Deeper": ModelConfig(
        model_class=TabBenchmarkResNet,
        model_params={**deepcopy(ResNet_params), **{"n_blocks": 8}},
        search_space=TabBenchmarkResNet.create_search_space()[0],
        default_values=[TabBenchmarkResNet.create_search_space()[1]],
    ),
    "TabBenchmarkResNet_Wider": ModelConfig(
        model_class=TabBenchmarkResNet,
        model_params={**deepcopy(ResNet_params), **{"blocks_dims": 512}},
        search_space=TabBenchmarkResNet.create_search_space()[0],
        default_values=[TabBenchmarkResNet.create_search_space()[1]],
    ),
    "TabBenchmarkMLP_GLU": ModelConfig(
        model_class=TabBenchmarkMLP,
        model_params=init_GLU_kwargs,
        search_space=TabBenchmarkMLP.create_search_space()[0],
        default_values=[TabBenchmarkMLP.create_search_space()[1]],
    ),
    "TabBenchmarkMLP_SNN": ModelConfig(
        model_class=TabBenchmarkMLP,
        model_params={
            **deepcopy(MLP_params),
            **dict(
                activation_fns=nn.SELU(),
                initialization_fns=init_snn,
                dropouts_modules_class=nn.AlphaDropout,
                norms_modules_class=nn.Identity,
            ),
        },
        search_space=TabBenchmarkMLP.create_search_space()[0],
        default_values=[TabBenchmarkMLP.create_search_space()[1]],
    ),
    "TabBenchmarkResNet_SNN": ModelConfig(
        model_class=TabBenchmarkResNet,
        model_params={
            **deepcopy(ResNet_params),
            **dict(
                activation_fns_1=nn.SELU(),
                activation_fns_2=nn.SELU(),
                initialization_fns_1=init_snn,
                initialization_fns_2=init_snn,
                dropouts_modules_class_1=nn.AlphaDropout,
                dropouts_modules_class_2=nn.AlphaDropout,
                norms_modules_class_1=nn.Identity,
                norms_modules_class_2=nn.Identity,
            ),
        },
        search_space=TabBenchmarkResNet.create_search_space()[0],
        default_values=[TabBenchmarkResNet.create_search_space()[1]],
    ),
    "TabBenchmarkResNet_GLU": ModelConfig(
        model_class=TabBenchmarkResNet,
        model_params=init_GLU_kwargs,
        search_space=TabBenchmarkResNet.create_search_space()[0],
        default_values=[TabBenchmarkResNet.create_search_space()[1]],
    ),
    "TabBenchmarkMLP_GReLUOneCycleLR": ModelConfig(
        model_class=TabBenchmarkMLP,
        model_params={
            **deepcopy(MLP_params),
            **dict(
                activation_fns=GeneralReLU(0.1, 0.4),
                initialization_fns=partial(nn.init.kaiming_normal_, a=0.1),
                lit_callbacks_tuples=[
                    (OneCycleLR, dict(max_lr=1e-3)),
                ],
                early_stopping_patience=0,
                max_epochs=200,
            ),
        },
        search_space=TabBenchmarkMLP.create_search_space()[0],
        default_values=[TabBenchmarkMLP.create_search_space()[1]],
    ),
    "TabBenchmarkResNet_GReLUOneCycleLR": ModelConfig(
        model_class=TabBenchmarkResNet,
        model_params={
            **deepcopy(ResNet_params),
            **dict(
                activation_fns_1=GeneralReLU(0.1, 0.4),
                activation_fns_2=GeneralReLU(0.1, 0.4),
                initialization_fns_1=partial(nn.init.kaiming_normal_, a=0.1),
                initialization_fns_2=partial(nn.init.kaiming_normal_, a=0.1),
                lit_callbacks_tuples=[
                    (OneCycleLR, dict(max_lr=1e-3)),
                ],
                early_stopping_patience=0,
                max_epochs=200,
            ),
        },
        search_space=TabBenchmarkResNet.create_search_space()[0],
        default_values=[TabBenchmarkResNet.create_search_space()[1]],
    ),
    "TabBenchmarkMLP_GReLUAutoOneCycleLR": ModelConfig(
        model_class=TabBenchmarkMLP,
        model_params={
            **deepcopy(MLP_params),
            **dict(
                activation_fns=GeneralReLU(0.1, 0.4),
                initialization_fns=partial(nn.init.kaiming_normal_, a=0.1),
                lit_callbacks_tuples=[
                    (AutomaticOneCycleLR, dict(suggestion_method="steep", early_stop_threshold=None)),
                ],
                early_stopping_patience=0,
                max_epochs=200,
            ),
        },
        search_space=TabBenchmarkMLP.create_search_space()[0],
        default_values=[TabBenchmarkMLP.create_search_space()[1]],
    ),
    "TabBenchmarkResNet_GReLUAutoOneCycleLR": ModelConfig(
        model_class=TabBenchmarkResNet,
        model_params={
            **deepcopy(ResNet_params),
            **dict(
                activation_fns_1=GeneralReLU(0.1, 0.4),
                activation_fns_2=GeneralReLU(0.1, 0.4),
                initialization_fns_1=partial(nn.init.kaiming_normal_, a=0.1),
                initialization_fns_2=partial(nn.init.kaiming_normal_, a=0.1),
                lit_callbacks_tuples=[
                    (AutomaticOneCycleLR, dict(suggestion_method="steep", early_stop_threshold=None)),
                ],
                early_stopping_patience=0,
                max_epochs=200,
            ),
        },
        search_space=TabBenchmarkResNet.create_search_space()[0],
        default_values=[TabBenchmarkResNet.create_search_space()[1]],
    ),
}

models_dict.update(models_dnns)
