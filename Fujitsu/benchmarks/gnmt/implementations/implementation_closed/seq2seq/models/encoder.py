import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import seq2seq.data.config as config
from seq2seq.utils import init_lstm_
import seq2seq.pack_utils._C as C


class Revert_varlen(torch.autograd.Function):
   @staticmethod
   def forward(ctx, input, offsets):
      ctx.offsets = offsets
      return C.revert_varlen_tensor(input, offsets)

   @staticmethod
   def backward(ctx, grad_output):
       return C.revert_varlen_tensor(grad_output, ctx.offsets), None

revert_varlen = Revert_varlen.apply

class EmuBidirLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, bias = True, batch_first = False, bidirectional = True):
        super(EmuBidirLSTM, self).__init__()
        assert num_layers == 1, "emulation bidirectional lstm works for a single layer only"
        assert batch_first == False, "emulation bidirectional lstm works for batch_first = False only"
        assert bidirectional == True, "use for bidirectional lstm only"
        self.bidir = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, bidirectional = True)
        self.layer1 = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first)
        self.layer2 = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first)
        self.layer1.weight_ih_l0 = self.bidir.weight_ih_l0
        self.layer1.weight_hh_l0 = self.bidir.weight_hh_l0
        self.layer2.weight_ih_l0 = self.bidir.weight_ih_l0_reverse
        self.layer2.weight_hh_l0 = self.bidir.weight_hh_l0_reverse
        self.layer1.bias_ih_l0 = self.bidir.bias_ih_l0
        self.layer1.bias_hh_l0 = self.bidir.bias_hh_l0
        self.layer2.bias_ih_l0 = self.bidir.bias_ih_l0_reverse
        self.layer2.bias_hh_l0 = self.bidir.bias_hh_l0_reverse

    @staticmethod
    def bidir_lstm(model, input, lengths):
        packed_input = pack_padded_sequence(input, lengths)
        out =  model(packed_input)[0]
        return pad_packed_sequence(out)[0]

    @staticmethod
    def emu_bidir_lstm(model0, model1, input, lengths):
        mask = C.set_mask_cpp(lengths).unsqueeze(-1).to(input.device,
            input.dtype, non_blocking = True)
        offsets = C.get_offsets(input, lengths)
        inputl1 = revert_varlen(input, offsets)
        out1 = model1(inputl1)
        outputs = revert_varlen(out1[0], offsets)
        out0 = model0(input)[0]*mask
        out_bi = torch.cat([out0, outputs], dim=2)
        return out_bi

    def forward(self, input, lengths):
        if (input.size(1) > 512):
            return self.bidir_lstm(self.bidir, input, lengths)
        else:
            return self.emu_bidir_lstm(self.layer2, self.layer1, input, lengths)


class ResidualRecurrentEncoder(nn.Module):
    """
    Encoder with Embedding, LSTM layers, residual connections and optional
    dropout.

    The first LSTM layer is bidirectional and uses variable sequence length
    API, the remaining (num_layers-1) layers are unidirectional. Residual
    connections are enabled after third LSTM layer, dropout is applied on
    inputs to LSTM layers.
    """
    def __init__(self, vocab_size, hidden_size=1024, num_layers=4, dropout=0.2,
                 batch_first=False, embedder=None, init_weight=0.1):
        """
        Constructor for the ResidualRecurrentEncoder.

        :param vocab_size: size of vocabulary
        :param hidden_size: hidden size for LSTM layers
        :param num_layers: number of LSTM layers, 1st layer is bidirectional
        :param dropout: probability of dropout (on input to LSTM layers)
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param embedder: instance of nn.Embedding, if None constructor will
            create new embedding layer
        :param init_weight: range for the uniform initializer
        """
        super(ResidualRecurrentEncoder, self).__init__()
        self.batch_first = batch_first
        self.rnn_layers = nn.ModuleList()
        # 1st LSTM layer, bidirectional
        self.rnn_layers.append(
            EmuBidirLSTM(hidden_size, hidden_size, num_layers=1, bias=True,
                         batch_first=batch_first, bidirectional=True))

        # 2nd LSTM layer, with 2x larger input_size
        self.rnn_layers.append(
            nn.LSTM((2 * hidden_size), hidden_size, num_layers=1, bias=True,
                    batch_first=batch_first))

        # Remaining LSTM layers
        for _ in range(num_layers - 2):
            self.rnn_layers.append(
                nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=True,
                        batch_first=batch_first))

        init_lstm_(self.rnn_layers[0].bidir)
        for lstm in self.rnn_layers[1:]:
            init_lstm_(lstm)

        self.dropout = nn.Dropout(p=dropout)

        self.share_embedding = (embedder is not None)
        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(vocab_size, hidden_size,
                                         padding_idx=config.PAD)
            nn.init.uniform_(self.embedder.weight.data, -init_weight, init_weight)

    def forward(self, inputs, lengths):
        """
        Execute the encoder.

        :param inputs: tensor with indices from the vocabulary
        :param lengths: vector with sequence lengths (excluding padding)

        returns: tensor with encoded sequences
        """
        if self.share_embedding and self.training:
            x = inputs
        else:
            x = self.embedder(inputs)

        # bidirectional layer
        x = self.dropout(x)
        x = self.rnn_layers[0](x, lengths)

        # 1st unidirectional layer
        x = self.dropout(x)
        x, _ = self.rnn_layers[1](x)

        # the rest of unidirectional layers,
        # with residual connections starting from 3rd layer
        for i in range(2, len(self.rnn_layers)):
            residual = x
            x = self.dropout(x)
            x, _ = self.rnn_layers[i](x)
            x = x + residual

        return x
