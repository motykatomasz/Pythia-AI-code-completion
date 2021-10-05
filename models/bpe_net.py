import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.code_completion_network_attention import Attention_custom


class lstm_encoder(nn.Module):
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
        super(lstm_encoder, self).__init__()
        self.padding_idx = padding_idx

        # TODO Why embedding for padding_idx changes slightly from 0 during training.
        # Embeddings
        # self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # LSTM settings
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=dropout)

        # Define the output layer
        self.output_dim = embedding_dim
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

        # # Project back to embedding space
        # self.decoder = nn.Linear(self.output_dim, vocab_size)
        # self.decoder.weight = self.embeddings.weight

    def forward(self, inputs, inputs_len):
        # embeds = self.embeddings(inputs)

        # Forward pass through LSTM layer
        # First deal with the padding
        embeds = pack_padded_sequence(inputs, inputs_len, batch_first=True, enforce_sorted=False)

        lstm_packed_out, (ht, _) = self.lstm(embeds)

        # Pad the sequence again to prevent memory leaks
        output, input_sizes = pad_packed_sequence(lstm_packed_out, padding_value=self.padding_idx)

        # Linear layer that learns projection matrix from dh dimensions to embedding dimensions (matrix A in the Paper)
        pred_embed = self.linear(ht[-1])

        return self.linear(output), pred_embed


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.emb_size = emb_size

        # changed GRU to LSTM
        self.lstm = nn.LSTM(self.emb_size + self.emb_size, self.emb_size, num_layers,
                          batch_first=True, dropout=dropout)

        # to initialize from the final encoder state
        self.bridge = nn.Linear(self.emb_size, self.emb_size, bias=True) if bridge else None
        self.log_softmax = nn.LogSoftmax(dim=2)

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size + emb_size,
                                          hidden_size, bias=False)

    # def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
    def forward_step(self, prev_embed, encoder_hidden, hidden, embeddings=None):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        if len(hidden.shape) < 3:
            query = hidden.unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        else:
            query = hidden.permute(1,0,2)

        # context = self.attention(query.squeeze(1), encoder_hidden, bpe=True)

        # update rnn hidden state
        # query is context for now (i.e. w/o attention)
        rnn_input = torch.cat([prev_embed, query], dim=2)
        output, (hidden, _) = self.lstm(rnn_input)

        # pre_output = torch.cat([prev_embed, output, context], dim=2)
        # pre_output = self.dropout_layer(pre_output)
        # pre_output = self.pre_output_layer(pre_output)

        # return output, hidden, pre_output
        return output, hidden

    # def forward(self, trg_embed, encoder_hidden, encoder_final,
    #             src_mask, trg_mask, hidden=None, max_len=None):

    def forward(self, target_embed, encoder_hidden, encoder_final, embeddings, hidden=None, max_len=20, testing=False):
        """Unroll the decoder one step at a time."""

        # max_len is the maximum number of steps to unroll the RNN
        # A hacky way of setting the max_len for now

        # initialize decoder hidden state
        if self.bridge is not None:
            hidden = self.init_hidden(encoder_final)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        if testing:
            # Just set to zeros as we concatenate it in the the forward step
            prev_embed = torch.zeros(target_embed[:, 0].unsqueeze(1).size()).cuda()

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):

            if not testing:
                # Allow it to look at 'i' previous tokens for generating a output at timestep i + 1
                prev_embed = target_embed[:, i].unsqueeze(1)

            output, hidden = self.forward_step(
                prev_embed, encoder_hidden, hidden, embeddings
            )
            # Allow it to see the previous output
            prev_embed = output
            output = torch.matmul(output, embeddings.weight.permute(1,0))
            # Normalize the output using log softmax
            output = self.log_softmax(output)
            decoder_states.append(output)
            # pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        # pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        # return decoder_states, hidden, pre_output_vectors  # [B, N, D]
        return decoder_states, hidden

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))


class BPE_net(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, embedding_dim, vocab_size, padding_idx, hidden_dim, batch_size, num_layers, dropout, max_label_len=20):
        super(BPE_net, self).__init__()
        self.encoder = lstm_encoder(embedding_dim, vocab_size,
                               padding_idx, hidden_dim,
                               batch_size, num_layers, dropout)
        # self.attention = Attention_custom(hidden_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, attention=Attention_custom(hidden_dim), num_layers=1, dropout=0, bridge=True)
        # self.src_embed = src_embed
        # self.trg_embed = trg_embed
        self.padding_idx = padding_idx
        self.max_label_len = max_label_len
        self.attention = Attention_custom(hidden_dim)

        # # encoder values
        # self.embedding_dim = embedding_dim
        # self.hidden_dim = hidden_dim
        # self.batch_size = batch_size
        # self.num_layers = num_layers

        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    # def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
    # target and target len will be available only while training
    def forward(self, inputs, inputs_len, target, target_len, testing=False):
        """Take in and process masked src and target sequences."""
        # we will use the hidden state, projected onto the embedding space, as the output of the LSTM encoder
        encoder_hidden, encoder_final = self.encode(inputs, inputs_len)
        # return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
        return self.decode(encoder_hidden, encoder_final, target, target_len, testing)

    def encode(self, inputs, inputs_len):
        # embed the inputs by using the embedding layer
        return self.encoder(self.embeddings(inputs), inputs_len)

    # def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
    #            decoder_hidden=None):
    #     return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
    #                         src_mask, trg_mask, hidden=decoder_hidden)

    def decode(self, encoder_hidden, encoder_final, target, target_len, testing, decoder_hidden=None):
        return self.decoder(self.embeddings(target), encoder_hidden, encoder_final, self.embeddings,
                            hidden=decoder_hidden, max_len=self.max_label_len, testing=testing)