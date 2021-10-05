import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Scaled dot-product attention: softmax(QK^transpose)/sqrt(n))*V
class Attention(nn.Module):
    def __init__(self, query_dim):
        super(Attention, self).__init__()
        # Scale according to hidden dimensions
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # Hidden Layer output from LST -> Query = [BxQ]
        # Output from LSTM -> Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = energy:[TxB], lin_comb:[BxV]

        # We assume q_dim == k_dim for dot product attention

        query = query.unsqueeze(1)  # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(0, 1).transpose(1, 2)  # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys)  # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = nn.functional.softmax(energy.mul_(self.scale), dim=2)  # scale, normalize

        values = values.transpose(0, 1)  # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1)  # [Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination


class Attention_custom(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention_custom, self).__init__()

        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.attn_bpe = nn.Linear(150*2, 150)
        self.v_bpe = nn.Parameter(torch.rand(150))
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, encoder_out, bpe=False):

        max_len = encoder_out.size(0)
        batch_size = encoder_out.size(1)

        H = inputs.repeat(max_len, 1, 1).transpose(0, 1)

        encoder_out = encoder_out.transpose(0, 1)  # [B*T*H]

        if bpe == True:
            attn_weights = self.score_bpe(H, encoder_out)
        else:
            attn_weights = self.score(H, encoder_out) # compute attention score

        attn_weights = nn.functional.softmax(attn_weights).unsqueeze(1)

        return attn_weights.bmm(encoder_out)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

    def score_bpe(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn_bpe(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v_bpe.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class NeuralCodeCompletionAttention(nn.Module):
    """
    Model definition for "PYTHIA: AI-ASSISTED CODE COMPLETION SYSTEM"

    Args:
        embedding_dim: The dimensionality of the token embedding.
        vocab_size: Size of the vocabulary.
        hidden_dim: The number of features in the hidden state `h`
        batch_size: Batch size
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
    """

    def __init__(self, embedding_dim, vocab_size, padding_idx, hidden_dim, batch_size, num_layers, dropout):
        super(NeuralCodeCompletionAttention, self).__init__()
        self.padding_idx = padding_idx

        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # LSTM settings
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)

        self.attention = Attention(self.hidden_dim)
        self.attention_custom = Attention_custom(self.hidden_dim)

        # Define the output layer
        self.output_dim = embedding_dim
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, inputs, inputs_len):
        embeds = self.embeddings(inputs)

        # Forward pass through LSTM layer
        # First deal with the padding
        embeds = pack_padded_sequence(embeds, inputs_len, batch_first=True, enforce_sorted=False)
        lstm_packed_out, (ht, ct) = self.lstm(embeds)

        # Pad the sequence again to prevent memory leaks
        unpacked_output, input_sizes = pad_packed_sequence(lstm_packed_out, padding_value=self.padding_idx)

        hidden = ht[-1]

        # SIMPLE SCALED ATTENTION:
        # The context is the output of the LSTM and since we're passing ht[-1] to the decoder, that is Query in our case.
        # attention(QUERY, KEY, VALUE) -> K=V
        # energy, linear_combination = self.attention(hidden, unpacked_output, unpacked_output)

        # ATTENTION with the learnable weights:
        linear_combination = self.attention_custom(hidden, unpacked_output)

        # Only take the output from the final timestep
        # Linear layer that learns projection matrix from dh dimensions to embedding dimensions (matrix A in the Paper)
        pred_embed = self.linear(linear_combination)

        # Projection back to vocabulary space
        projection = torch.matmul(self.embeddings.weight, pred_embed.squeeze(1).permute(1, 0)).permute(1, 0)

        return projection
