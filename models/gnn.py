import torch
import torch.nn as nn
from random import random
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hid_dim, improved=True)
        self.conv2 = GCNConv(hid_dim, hid_dim * 2, improved=True)
        self.conv3 = GCNConv(hid_dim * 2, hid_dim, improved=True)
        # self.conv4 = GCNConv(hid_dim * 4, hid_dim * 2, improved=True)
        # self.conv5 = GCNConv(hid_dim * 2, hid_dim, improved=True)
        self.drop1 = nn.Dropout(0.9)

    def forward(self, data, edge_index, batch):
        # batch = [num node * num graphs in batch]
        # data = [num node * num graphs in batch, num features]
        # edge_index = [2, num edges * num graph in batch]
        # Obtain node embeddings 
        # data = [num node * num graphs in batch, hid dim]
        output = self.conv1(data, edge_index)
        output = output.relu()
        output = self.conv2(output, edge_index)
        output = output.relu()
        output = self.conv3(output, edge_index)
        # output = output.relu()
        # output = self.conv4(output, edge_index)
        # output = output.relu()
        # output = self.conv5(output, edge_index)

        # Readout layer
        # output = [batch size, hid dim]
        output = global_max_pool(output, batch)
        
        return output

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data, hidden, context):
        # data = [batch size]
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim] 
        data = data.unsqueeze(0)
        # embedded = [1, batch size, hid dim]
        embedded = self.dropout(self.embedding(data))
        # emb_con = [1, batch size, emb dim + hid dim]
        emb_con = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(emb_con, hidden)
        # output = [batch size, emb dim + hid dim * 2]
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)
        # prediction = [batch size, output dim]
        prediction = self.fc_out(output)

        return prediction, hidden

class Graph2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, data, teacher_forcing_ratio=0.5):
        # tgt = [1, output_dim * batch_size] -> [output_dim, batch_size], float -> long
        # batch_size = num of graphs in this batch
        tgt = torch.reshape(data.y, (data.num_graphs, -1)).t()
        # tgt_len is max target expression length.
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(tgt_len, data.num_graphs, tgt_vocab_size).to(self.device)
        # context = [1, batch_size, hid_dim]
        context = self.encoder(data.x, data.edge_index, data.batch).unsqueeze(0)

        hidden = context

        expr = tgt[0,:]
        for t in range(1, tgt_len):
            # expr = [batch_size]
            # hidden = [1, batch_size, hid_dim]
            # context = [1, batch_size, hid_dim]
            output, hidden = self.decoder(expr, hidden, context)
            outputs[t] = output
            teacher_force = random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            expr = tgt[t] if teacher_forcing_ratio else top1
        
        return outputs
