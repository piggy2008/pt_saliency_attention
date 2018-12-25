import torch.nn as nn
from torch.autograd import Variable
import torch

import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, normalize=True):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.normalize = normalize

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        if self.normalize:
            self.g_norm = nn.LayerNorm([self.hidden_dim, self.height, self.width])
            self.i_norm = nn.LayerNorm([self.hidden_dim, self.height, self.width])
            self.f_norm = nn.LayerNorm([self.hidden_dim, self.height, self.width])

            self.o_norm = nn.LayerNorm([self.hidden_dim, self.height, self.width])
            self.c_norm = nn.LayerNorm([self.hidden_dim, self.height, self.width])

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        if self.normalize:
            cc_i = self.i_norm(cc_i)
            cc_g = self.g_norm(cc_g)
            cc_f = self.f_norm(cc_f)

        i = F.sigmoid(cc_i)
        f = F.sigmoid(cc_f + 1.0)
        g = F.tanh(cc_g)

        c_next = f * c_cur + i * g

        if self.normalize:
            cc_o = self.g_norm(cc_o)
            c_next = self.f_norm(c_next)

        o = F.sigmoid(cc_o)
        h_next = o * F.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class ConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

        ###### history hidden state #######
        self.w_h = nn.Conv2d(in_channels=self.hidden_dim[0], out_channels=1, kernel_size=3, padding=1)
        self.w_x = nn.Conv2d(in_channels=self.input_dim, out_channels=1, kernel_size=3, padding=1)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            h_history = []
            c_history = []
            input_list = []
            for t in range(seq_len):
                if t > 1:
                    h, c = self._hidden_attention(h_history, input_list)
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
                h_history.append(h)
                c_history.append(c)
                input_list.append(cur_layer_input[:, t, :, :, :])

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    def _hidden_attention(self, h_history, c_history, input_list):
        a_history = []
        for i, h in enumerate(h_history):
            a_h = F.adaptive_avg_pool2d(F.tanh(self.w_h(h) + self.w_x(input_list[i])), output_size=(1, 1))
            a_h = a_h.reshape([a_h.size(0), -1])
            a_history.append(a_h)

        a_history_cat = torch.cat(a_history, dim=1)
        attention = F.softmax(a_history_cat, dim=1)
        attention = attention.chunk(len(h_history), dim=1)
        h_atte = torch.zeros_like(h_history[0])
        c_atte = torch.zeros_like(c_history[0])
        for i, h in enumerate(h_history):
            h_atte += attention[i] * h

        for i, c in enumerate(c_history):
            c_atte += attention[i] * c

        return h_atte, c_atte

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param