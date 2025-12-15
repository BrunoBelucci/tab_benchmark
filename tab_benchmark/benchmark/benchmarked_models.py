from functools import partial
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

models_dict = {model_cls.__name__: (model_cls, {}) for model_cls in all_sklearn_models}
# enable probability in TabBenchmarkNuSVC
models_dict['TabBenchmarkNuSVC'] = (models_dict['TabBenchmarkNuSVC'][0], {'probability': True})

models_dict.update({
    TabBenchmarkXGBClassifier.__name__: (TabBenchmarkXGBClassifier, TabBenchmarkXGBClassifier.get_recommended_params()),
    TabBenchmarkXGBRegressor.__name__: (TabBenchmarkXGBRegressor, TabBenchmarkXGBRegressor.get_recommended_params()),
    TabBenchmarkLGBMClassifier.__name__: (
        TabBenchmarkLGBMClassifier, TabBenchmarkLGBMClassifier.get_recommended_params()),
    TabBenchmarkLGBMRegressor.__name__: (TabBenchmarkLGBMRegressor, TabBenchmarkLGBMRegressor.get_recommended_params()),
    TabBenchmarkCatBoostClassifier.__name__: (TabBenchmarkCatBoostClassifier,
                                              TabBenchmarkCatBoostClassifier.get_recommended_params()),
    TabBenchmarkCatBoostRegressor.__name__: (TabBenchmarkCatBoostRegressor,
                                             TabBenchmarkCatBoostRegressor.get_recommended_params()),
})

MLP_recommended_params = TabBenchmarkMLP.get_recommended_params()
ResNet_recommended_params = TabBenchmarkResNet.get_recommended_params()
Transformer_recommended_params = TabBenchmarkTransformer.get_recommended_params()
Saint_recommended_params = TabBenchmarkSaint.get_recommended_params()
Node_recommended_params = TabBenchmarkNode.get_recommended_params()
TabNet_recommended_params = TabBenchmarkTabNet.get_recommended_params()
TabTransformer_recommended_params = TabBenchmarkTabTransformer.get_recommended_params()
MLP_params = MLP_recommended_params.copy()
ResNet_params = ResNet_recommended_params.copy()
Transformer_params = Transformer_recommended_params.copy()
Saint_params = Saint_recommended_params.copy()
Node_params = Node_recommended_params.copy()
TabNet_params = TabNet_recommended_params.copy()


def init_GLU_kwargs(model_cls, **kwargs):
    # lazy initialization
    initialization_fn = partial(initialize_glu_, input_dim=256, output_dim=256)
    initialization_fn.__name__ = 'initialize_glu_'
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


models_dict.update(
    {
        'TabBenchmarkMLP': (TabBenchmarkMLP, MLP_params.copy()),
        'TabBenchmarkResNet': (TabBenchmarkResNet, ResNet_params.copy()),
        'TabBenchmarkTransformer': (TabBenchmarkTransformer, Transformer_params.copy()),
        'TabBenchmarkSaint': (TabBenchmarkSaint, Saint_params.copy()),
        'TabBenchmarkNode': (TabBenchmarkNode, Node_params.copy()),
        'TabBenchmarkTabNet': (TabBenchmarkTabNet, TabNet_params.copy()),
        'TabBenchmarkTabTransformer': (TabBenchmarkTabTransformer, TabTransformer_recommended_params.copy()),
        'TabBenchmarkMLP_Deeper': (TabBenchmarkMLP, {**MLP_params.copy(), **{'n_layers': 8}}),
        'TabBenchmarkMLP_Wider': (TabBenchmarkMLP, {**MLP_params.copy(), **{'hidden_dims': 512}}),
        'TabBenchmarkResNet_Deeper': (TabBenchmarkResNet, {**ResNet_params.copy(), **{'n_blocks': 8}}),
        'TabBenchmarkResNet_Wider': (TabBenchmarkResNet, {**ResNet_params.copy(), **{'blocks_dims': 512}}),
        'TabBenchmarkMLP_GLU': (TabBenchmarkMLP, init_GLU_kwargs),
        'TabBenchmarkMLP_SNN': (
            TabBenchmarkMLP, {**MLP_params.copy(), **dict(activation_fns=nn.SELU(), initialization_fns=init_snn,
                                                          dropouts_modules_class=nn.AlphaDropout,
                                                          norms_modules_class=nn.Identity)}),
        'TabBenchmarkResNet_SNN': (
            TabBenchmarkResNet, {**ResNet_params.copy(), **dict(activation_fns_1=nn.SELU(), activation_fns_2=nn.SELU(),
                                                                initialization_fns_1=init_snn,
                                                                initialization_fns_2=init_snn,
                                                                dropouts_modules_class_1=nn.AlphaDropout,
                                                                dropouts_modules_class_2=nn.AlphaDropout,
                                                                norms_modules_class_1=nn.Identity,
                                                                norms_modules_class_2=nn.Identity)}),
        'TabBenchmarkResNet_GLU': (TabBenchmarkResNet, init_GLU_kwargs),
        'TabBenchmarkMLP_GReLUOneCycleLR': (
            TabBenchmarkMLP, {**MLP_params.copy(), **dict(activation_fns=GeneralReLU(0.1, 0.4),
                                                          initialization_fns=partial(nn.init.kaiming_normal_, a=0.1),
                                                          lit_callbacks_tuples=[(OneCycleLR, dict(max_lr=1e-3)), ],
                                                          early_stopping_patience=0, max_epochs=200)}),
        'TabBenchmarkResNet_GReLUOneCycleLR': (
            TabBenchmarkResNet, {**ResNet_params.copy(),
                                 **dict(activation_fns_1=GeneralReLU(0.1, 0.4), activation_fns_2=GeneralReLU(0.1, 0.4),
                                        initialization_fns_1=partial(nn.init.kaiming_normal_, a=0.1),
                                        initialization_fns_2=partial(nn.init.kaiming_normal_, a=0.1),
                                        lit_callbacks_tuples=[(OneCycleLR, dict(max_lr=1e-3)), ],
                                        early_stopping_patience=0, max_epochs=200)}),
        'TabBenchmarkMLP_GReLUAutoOneCycleLR': (
            TabBenchmarkMLP, {**MLP_params.copy(), **dict(activation_fns=GeneralReLU(0.1, 0.4),
                                                          initialization_fns=partial(nn.init.kaiming_normal_, a=0.1),
                                                          lit_callbacks_tuples=[
                                                              (AutomaticOneCycleLR, dict(suggestion_method='steep',
                                                                                         early_stop_threshold=None)), ],
                                                          early_stopping_patience=0, max_epochs=200)}),
        'TabBenchmarkResNet_GReLUAutoOneCycleLR': (
            TabBenchmarkResNet, {**ResNet_params.copy(),
                                 **dict(activation_fns_1=GeneralReLU(0.1, 0.4), activation_fns_2=GeneralReLU(0.1, 0.4),
                                        initialization_fns_1=partial(nn.init.kaiming_normal_, a=0.1),
                                        initialization_fns_2=partial(nn.init.kaiming_normal_, a=0.1),
                                        lit_callbacks_tuples=[(AutomaticOneCycleLR, dict(suggestion_method='steep',
                                                                                         early_stop_threshold=None)), ],
                                        early_stopping_patience=0, max_epochs=200)}),
    }
)
