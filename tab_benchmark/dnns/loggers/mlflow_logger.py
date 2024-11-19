from argparse import Namespace
from typing import Union, Dict, Any

from lightning.pytorch.loggers import MLFlowLogger as OriginalMLFlowLogger
from lightning.fabric.utilities.logger import _convert_params, _flatten_dict


class MLFlowLogger(OriginalMLFlowLogger):
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        # Sanitize callables
        for param in params.keys():
            if callable(params[param]):
                try:
                    params[param] = params[param].__name__
                except AttributeError:
                    params[param] = str(params[param])
        super().log_hyperparams(params)
