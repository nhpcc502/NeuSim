import torch
import torch.nn as nn
from random import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)    
        self.rnn = nn.GRU(emb_dim, hid_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src_len, batch_size, emb_dim]
        outputs, hidden = self.rnn(embedded)
        # outputs = [src_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_directions, batch_size, hid_dim]
        # cell = [n_layers * n_directions, batch_size, hid_dim]
        # outputs are always from the top hidden layer

        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.output_dim = output_dim        
        self.embedding = nn.Embedding(output_dim, emb_dim)    
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, dropout=dropout)        
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data, hidden, context):        
        # data = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]        
        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]        
        data = data.unsqueeze(0)
        # data = [1, batch size]        
        embedded = self.dropout(self.embedding(data))        
        # embedded = [1, batch size, emb dim]                
        emb_con = torch.cat((embedded, context), dim=2)       
        # emb_con = [1, batch size, emb dim + hid dim]         
        output, hidden = self.rnn(emb_con, hidden)        
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]        
        # seq len, n layers and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]        
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)        
        # output = [batch size, emb dim + hid dim * 2]        
        prediction = self.fc_out(output)        
        # prediction = [batch size, output dim]

        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5): 
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim        
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)        
        # last hidden state of the encoder is the context
        context = self.encoder(src)        
        # context also used as the initial hidden state of the decoder
        hidden = context        
        # first input to the decoder is the <sos> tokens
        data = trg[0,:]  
              
        for t in range(1, trg_len):            
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(data, hidden, context)            
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output            
            # decide if we are going to use teacher forcing or not
            teacher_force = random() < teacher_forcing_ratio            
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)            
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            data = trg[t] if teacher_force else top1

        return outputs
