from IPython.core.display_functions import display
from tabular_benchmark.dnns.callbacks import DefaultLogs
import torch
import matplotlib.pyplot as plt


class PlotLoss(DefaultLogs):
    def __init__(self, plot_every_n_epochs=1):
        super().__init__()
        self.graph_out = None
        self.ax = None
        self.fig = None
        self.plot_every_n_epochs = plot_every_n_epochs
        self.train_epoch_losses = {}
        self.validation_epoch_losses = {}

    def on_epoch_end(self, trainer, pl_module, str_set):
        if not trainer.sanity_checking:
            step_losses_dict = getattr(self, str_set + '_step_losses')
            epoch_losses_dict = getattr(self, str_set + '_epoch_losses')
            for dataloader_idx in step_losses_dict:
                losses = step_losses_dict[dataloader_idx]
                mean_loss = torch.tensor(losses).mean()
                if dataloader_idx not in epoch_losses_dict:
                    epoch_losses_dict[dataloader_idx] = []
                epoch_losses_dict[dataloader_idx].append(mean_loss)
                step_losses_dict[dataloader_idx].clear()

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self.on_epoch_end(trainer, pl_module, 'train')
            if self.graph_out is None:
                self.fig, self.ax = plt.subplots()
                self.graph_out = display(self.ax.figure, display_id=True)
            if trainer.current_epoch % self.plot_every_n_epochs == 0:
                self.ax.clear()
                for dataloader_idx in self.train_epoch_losses:
                    self.ax.plot(self.train_epoch_losses[dataloader_idx], label=f'train_{dataloader_idx}')
                for dataloader_idx in self.validation_epoch_losses:
                    self.ax.plot(self.validation_epoch_losses[dataloader_idx], label=f'validation_{dataloader_idx}')
                self.ax.legend()
                self.graph_out.update(self.ax.figure)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module, 'validation')

    def on_test_epoch_end(self, trainer, pl_module):
        pass

    def on_fit_end(self, trainer, pl_module):
        plt.close(self.fig)  # close the figure at the end of the training to avoid displaying it twice
