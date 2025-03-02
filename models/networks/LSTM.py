from typing import Dict

import torch
import torch.nn as nn


class LSTM(nn.Module):
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'lstm', 'head']

    def __init__(self, in_features: int, hidden_size, num_layers: int = 1):
        super().__init__()

        self.embedding_net = nn.Linear(51, 10) # fully connected layer - possible bridge for missing values

        self.lstm = nn.LSTM(input_size=24, hidden_size=256, num_layers=1)

        # self.dropout = nn.Dropout(p=cfg.output_dropout) # default 0?

        self.head = nn.Linear(hidden_size, 1)


    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the CudaLSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [batch size, 1, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [batch size, 1, hidden size].
        """
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data)
        lstm_output, (h_n, c_n) = self.lstm(input=x_d)

        # reshape to [batch_size, seq, n_hiddens]
        lstm_output = lstm_output.transpose(0, 1)
        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)

        pred = {'lstm_output': lstm_output, 'h_n': h_n, 'c_n': c_n}
        pred.update(self.head(self.dropout(lstm_output)))

        return pred
