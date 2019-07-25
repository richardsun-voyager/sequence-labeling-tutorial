import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.nn import utils as nn_utils


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, batch_first=True,
                            num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
    

    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2),
                torch.randn(2, batch_size, self.hidden_dim // 2))
    
    def forward(self, sentences, seq_lens):
        '''
        Args:
            sentences: batch_size*word_num, long tensor
            seq_lens: batch_size, long tensor
        Return:
            hidden states for each sequence of tokens
        '''
        batch_size, word_num = sentences.size()
        
        self.hidden = self.init_hidden(batch_size)
        embeds = self.word_embeds(sentences)#batch_size*word_num*emb_dim
        embeds = self.dropout(embeds)
        
        #Sorting according to the lengths
        perm_seq_lens, perm_idx = seq_lens.sort(0, descending=True)
        _, desorted_perm_idx = torch.sort(perm_idx, descending=False)
        embeds = embeds[perm_idx]
        pack_embeds = nn_utils.rnn.pack_padded_sequence(embeds, 
                                                 perm_seq_lens, batch_first=True)
        
        lstm_out, self.hidden = self.lstm(pack_embeds, self.hidden)
        lstm_out, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        #Restore the order of sentences
        return lstm_out[desorted_perm_idx]