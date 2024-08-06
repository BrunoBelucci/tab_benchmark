from tab_benchmark.dnns.modules.tabular_module import TabularModule


class TabNetModule(TabularModule):
    """LightningModule exclusively for TabNet.

    The base step is slightly different to allow for the M_loss to be subtracted from the loss.
    """
    def base_step(self, batch, batch_idx):
        y_true = batch['y']
        model_output = self.model(batch)
        y_pred = model_output['y_pred']
        M_loss = model_output['M_loss']
        loss = self.loss_fn(y_pred, y_true)
        loss = loss - M_loss  # already multiplied by lambda_sparse in model
        outputs = {
            'loss': loss,
            'y_pred': y_pred,
            'M_loss': M_loss,
        }
        return outputs

