# from tabular_benchmark.dnns.modules.tabular_module import TabularModule
#
#
# class RidgeGDModule(TabularModule):
#     """LightningModule exclusively for RidgeGD.
#
#     We add the on_fit_start method to cache the values for the model (calculate covariance matrix by batch).
#     """
#     def on_fit_start(self):
#         trainer = self.trainer
#         for batch in trainer.datamodule.train_dataloader():
#             for key, tensor in batch.items():
#                 batch[key] = tensor.to(device=self.device)
#             self.model.pre_forward(batch)
#         self.model.cache_values()
