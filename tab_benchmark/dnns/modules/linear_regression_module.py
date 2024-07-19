# from tabular_benchmark.dnns.modules.ridge_gd_module import RidgeGDModule
#
#
# class LinearRegressionModule(RidgeGDModule):
#     """LightningModule exclusively for LinearRegression.
#
#     We just disable automatic optimization, as a linear regression model does not need it.
#     """
#     def __init__(self, model_class, architecture_kwargs, optimizer_fn, optimizer_kwargs, loss_fn,
#                  scheduler_fn=None, scheduler_kwargs=None, scheduler_config=None):
#         super().__init__(dnn_architecture_class=model_class, architecture_kwargs=architecture_kwargs,
#                          torch_optimizer_fn=optimizer_fn, torch_optimizer_kwargs=optimizer_kwargs, loss_fn=loss_fn,
#                          torch_scheduler_fn=scheduler_fn, torch_scheduler_kwargs=scheduler_kwargs,
#                          lit_scheduler_config=scheduler_config)
#         self.automatic_optimization = False
