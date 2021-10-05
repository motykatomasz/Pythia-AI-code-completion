import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NeuralCodeCempletion(nn.Module):
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
        super(NeuralCodeCempletion, self).__init__()
        self.padding_idx = padding_idx

        # TODO Why embedding for padding_idx changes slightly from 0 during training.
        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

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

        # Project back to embedding space
        self.decoder = nn.Linear(self.output_dim, vocab_size)
        self.decoder.weight = self.embeddings.weight

    def forward(self, inputs, inputs_len):
        embeds = self.embeddings(inputs)

        # Forward pass through LSTM layer
        # First deal with the padding
        embeds = pack_padded_sequence(embeds, inputs_len, batch_first=True, enforce_sorted=False)

        lstm_packed_out, (ht, _) = self.lstm(embeds)

        # Pad the sequence again to prevent memory leaks
        unpacked_output, input_sizes = pad_packed_sequence(lstm_packed_out, padding_value=self.padding_idx)

        # Only take the output from the final timestep
        # Linear layer that learns projection matrix from dh dimensions to embedding dimensions (matrix A in the Paper)
        pred_embed = self.linear(ht[-1])

        # Projection back to vocabulary space
        #projection = torch.matmul(self.embeddings.weight,  pred_embed.permute(1, 0)).permute(1, 0)
        projection = self.decoder(pred_embed)

        return projection

    def beam_search_forward(self, inputs, inputs_len, vocab, beam_width=3):
        from scipy.special import softmax
        import numpy as np
        from queue import PriorityQueue
        from functools import total_ordering

        @total_ordering
        class Sample:
            def __init__(self, s: str, indices, logp: float, state):
                self.s = s
                self.indices = indices
                self.logp = logp
                self.state = state
                
            def __lt__(self, other):
                self.logp < other.logp

            def __eq__(self, other):
                self.logp == other.logp

        beam_width = 3
        topk = 10
        decoded_batch = []

        inputs.detach()
        for i in range(inputs.size(0)):
            embeds = self.embeddings(inputs[i:i+1])
            embeds = pack_padded_sequence(embeds, inputs_len[i:i+1], batch_first=True, enforce_sorted=False)

            _, (h0, c0) = self.lstm(embeds)
            top_10 = PriorityQueue(topk)
            candidates = PriorityQueue()
            pred_embed = self.linear(h0[-1])
            projection = self.decoder(pred_embed)
            probs = np.log(softmax(projection.detach().numpy().flatten()))
            best = np.argsort(probs)[::-1]
            for _, i in zip(range(beam_width), best):
                candidates.put(Sample(str(vocab[i]), [i], probs[i], (h0, c0)), block=False)

            while not candidates.empty():
                candidate = candidates.get(block=False)
                embed = self.embeddings(torch.Tensor([[candidate.indices[-1]]]).to(torch.long))
                lstm_packed_out, (ht, ct) = self.lstm(embed, candidate.state)
                pred_embed = self.linear(ht[-1])
                projection = self.decoder(pred_embed)
                probs = np.log(softmax(projection.detach().numpy().flatten()))
                best = np.argsort(probs)[::-1]
                for _, i in zip(range(beam_width), best):
                    candidates.put(Sample(candidate.s + " " + str(vocab[i]), candidate.indices + [i], candidate.logp + probs[i], (ht, ct)), block=False)
                    print(candidate.s + " " + str(vocab[i]))
