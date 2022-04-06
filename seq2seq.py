import argparse
from http.client import BadStatusLine
from imghdr import tests
import torch
import torch.optim as optim
import torch.nn as nn
import z3
import warnings

from time import time
from tqdm import tqdm

from torchtext.legacy.data import Field, BucketIterator, TabularDataset


warnings.filterwarnings("ignore")

class Dataset:
    def __init__(self, model_name, dataset):
        self.model_name = model_name
        self.dataset = dataset
        self._build_vocab()

    def _tokenize_reverse(self, text):
        return [char for char in text[::-1]]
    
    def _tokenize(self, text):
        return [char for char in text]
    
    def _build_vocab(self):
        self.SRC = Field(
            tokenize=self._tokenize_reverse if self.model_name=='lstm' else self._tokenize,
            # tokenize=self._tokenize_reverse,
            init_token='<sos>', 
            eos_token='<eos>',
            pad_token='<pad>',
            lower=True if self.model_name=='bert' else False,
            batch_first=True if self.model_name=='bert' else False
        )
        self.TGT = Field(
            tokenize=self._tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            pad_token='<pad>',
            lower=True if self.model_name=='bert' else False,
            batch_first=True if self.model_name=='bert' else False
        )

    def build_dataset(self):
        expr = TabularDataset(
            path=f'dataset/raw/{self.dataset}.csv',
            format='csv',
            fields=[
                ('src', self.SRC),
                ('tgt', self.TGT)
            ],
        )

        train_data, valid_data = expr.split(split_ratio=0.9)
        self.SRC.build_vocab(train_data)
        self.TGT.build_vocab(train_data)

        print()
        print(f'Dataset: {self.dataset}')
        print('=======================================')
        print(f'Train samples: {len(train_data)}')
        print(f'Valid samples: {len(valid_data)}')

        return train_data, valid_data

class Model:
    def __init__(self, model_name, input_dim, output_dim, src_pad_idx, tgt_pad_idx):
        self.model_name = model_name
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _count_paras(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def build_model(self):
        emb_dim = 256
        hid_dim = 512
        dropout = 0.1 if self.model_name=='bert' else 0.1
        device = torch.device('cuda')

        if self.model_name == 'gru':
            from models.gru import Encoder, Decoder, Seq2Seq
            enc = Encoder(self.input_dim, emb_dim, hid_dim, dropout)
            dec = Decoder(self.output_dim, emb_dim, hid_dim, dropout)
            model = Seq2Seq(enc, dec, device).to(device)
        elif self.model_name == 'bi_gru':
            from models.bi_gru import Encoder, Decoder, Attention, Seq2Seq
            attn = Attention(hid_dim)
            enc = Encoder(self.input_dim, emb_dim, hid_dim, dropout)
            dec = Decoder(self.output_dim, emb_dim, hid_dim, dropout, attn)
            model = Seq2Seq(enc, dec, device).to(device)
        elif self.model_name == 'lstm':
            from models.lstm import Encoder, Decoder, Seq2Seq
            layers = 2
            enc = Encoder(self.input_dim, emb_dim, hid_dim, layers, dropout)
            dec = Decoder(self.output_dim, emb_dim, hid_dim, layers, dropout)
            model = Seq2Seq(enc, dec, device).to(device)
        elif self.model_name == 'bert':
            from models.bert import Encoder, Decoder, Seq2Seq
            layers = 3
            heads = 8
            hid_dim = 256
            pf_dim = 512
            enc = Encoder(self.input_dim, hid_dim, layers, heads, pf_dim, dropout, device)
            dec = Decoder(self.output_dim, hid_dim, layers, heads, pf_dim, dropout, device)
            model = Seq2Seq(enc, dec, src_pad_idx, tgt_pad_idx, device).to(device)
        else:
            raise NotImplementedError

        print()
        print(f'Modle: {self.model_name}')
        print('=======================================')
        print(f'Input dim:  {self.input_dim}')
        print(f'output dim: {self.output_dim}')
        print(f'Hidden dim: {hid_dim}')
        print(f'Paras:      {self._count_paras(model):,}')

        return model

class Verifer:
    def __init__(self, src_field, tgt_field, model, model_name):
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.model = model
        self.model_name = model_name
        self.device = torch.device('cuda')

    def translate(self, expr, max_len=110):
        self.model.eval()
            
        tokens = [token.lower() for token in expr]
        tokens = [self.src_field.init_token] + tokens + [self.src_field.eos_token]
            
        src_indexes = [self.src_field.vocab.stoi[token] for token in tokens]    
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0 if self.model_name=='bert' else 1).to(self.device)
        if self.model_name == 'bert':
            src_mask = model.make_src_mask(src_tensor)
        
        with torch.no_grad():
            if self.model_name == 'gru':
                context = self.model.encoder(src_tensor)
            elif self.model_name == 'lstm':
                hidden, context = self.model.encoder(src_tensor)
            elif self.model_name == 'bert':
                enc_src = model.encoder(src_tensor, src_mask)
        if self.model_name == 'gru':
            hidden = context
        tgt_indexes = [self.tgt_field.vocab.stoi[self.tgt_field.init_token]]

        for i in range(max_len):
            if self.model_name != 'bert':
                tgt_tensor = torch.LongTensor([tgt_indexes[-1]]).to(self.device)
            else:
                tgt_tensor = torch.LongTensor(tgt_indexes).unsqueeze(0).to(self.device)
                tgt_mask = model.make_tgt_mask(tgt_tensor)

            with torch.no_grad():
                if self.model_name == 'gru':
                    output, hidden = self.model.decoder(tgt_tensor, hidden, context)
                elif self.model_name == 'lstm':
                    output, hidden, context = model.decoder(tgt_tensor, hidden, context)
                elif self.model_name == 'bert':
                    output, attention = model.decoder(tgt_tensor, enc_src, tgt_mask, src_mask)
            
            if self.model_name != 'bert':
                pred_token = output.argmax(1).item()        
            else:
                pred_token = output.argmax(2)[:,-1].item()
            tgt_indexes.append(pred_token)

            if pred_token == self.tgt_field.vocab.stoi[self.tgt_field.eos_token]:
                break
        
        tgt_tokens = [self.tgt_field.vocab.itos[i] for i in tgt_indexes]
        
        return ''.join(tgt_tokens[1:-1])

    def z3_verify(self, src, tgt):
        x, y, z, t = z3.BitVecs('x y z t', 4)
        s = z3.Solver()
        try:
            s.add(eval(src) != eval(tgt))
        except:
            return False
        if s.check() == z3.unsat:
            return True
        return False

    def count_acc(self, testset):
        print()
        print('Verification...')
        print('=======================================')

        num_total_equal, num_seman_equal = 0, 0
        total_infer_time = 0

        for idx in tqdm(range(len(testset))):
            src = ''.join(vars(testset.examples[idx])['src'])
            tgt = ''.join(vars(testset.examples[idx])['tgt'])

            start_time = time()
            pred = self.translate(src)
            total_infer_time += time() - start_time
            print(f'Src:  {src}')
            print(f'Tgt:  {tgt}')
            print(f'Pred: {pred}')
            
            if pred == tgt:
                num_total_equal += 1
            elif self.z3_verify(pred, tgt):
                num_seman_equal += 1

        size = len(testset)
        print(f'Validation set size: {size}')
        print(f'Inference time per sample: {total_infer_time/size:.4f}')
        print(f'Formal equal count:\t{num_total_equal}/{size}')
        print(f'Semantic equal count:\t{num_seman_equal}/{size}')
        print(f'Accuracy:\n\tWithout semantic equal:\t{num_total_equal/size:.4f}')
        print(f'\tWith semantic equal:\t{(num_total_equal + num_seman_equal)/size:.4f}')


class Trainer:
    def __init__(self, model, epochs, pad_token, model_name):
        self.model = model
        self.epochs = epochs
        self.device = torch.device('cuda')
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=10, verbose=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
        self.model_name = model_name
        
    def train(self, iterator):    
        self.model.train()    
        epoch_loss = 0
        
        for i, batch in enumerate(iterator):
            src = batch.src
            tgt = batch.tgt        
            self.optimizer.zero_grad()

            if self.model_name != 'bert':
                output = self.model(src, tgt)
            else:
                output, _ = self.model(src, tgt[:,:-1])

            #tgt = [tgt len, batch size]
            #output = [tgt len, batch size, output dim]
            output_dim = output.shape[-1]

            if self.model_name != 'bert':
                output = output[1:].view(-1, output_dim)
                tgt = tgt[1:].view(-1)
            else:
                output = output.contiguous().view(-1, output_dim)
                tgt = tgt[:,1:].contiguous().view(-1)

            #tgt = [(tgt len - 1) * batch size]
            #output = [(tgt len - 1) * batch size, output dim]
                 
            loss = self.criterion(output, tgt)
            loss.backward()
            # clip equal to 1

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)

    def evaluate(self, iterator):
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch.src
                tgt = batch.tgt
                if self.model_name != 'bert':
                    output = self.model(src, tgt, 0) #turn off teacher forcing
                else:
                    output, _ = self.model(src, tgt[:,:-1])
                #tgt = [tgt len, batch size]
                #output = [tgt len, batch size, output dim]

                output_dim = output.shape[-1]

                if self.model_name != 'bert':
                    output = output[1:].view(-1, output_dim)
                    tgt = tgt[1:].view(-1)
                else:
                    output = output.contiguous().view(-1, output_dim)
                    tgt = tgt[:,1:].contiguous().view(-1)
                #tgt = [(tgt len - 1) * batch size]
                #output = [(tgt len - 1) * batch size, output dim]
                loss = self.criterion(output, tgt)

                epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)

    def run(self, model, train_iter, valid_iter, model_name):
        print()
        print('Training...')
        print('=======================================')
        best_valid_loss = float('inf')
        best_train_loss = float('inf')
        
        # param used in early stoping
        es_param = 'valid_loss'
        
        # patience for early stopping
        patience = 20
        trigger = 0
        total_train_time = 0

        for epoch in range(self.epochs):
            start_time = time()
            train_loss = self.train(train_iter)
            total_train_time += time() - start_time
            valid_loss = self.evaluate(valid_iter)
            print(f'Epoch {epoch+1:>3d}: train loss: {train_loss:.4f}, val loss: {valid_loss:.4f}, time: {time() - start_time:.4f}')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                trigger = 0
                torch.save(model.state_dict(), f'saved_models/{model_name}.pt')
            else:
                trigger += 1
                if trigger >= patience:
                    break
            self.scheduler.step(valid_loss)
        print(f'Best valid loss: {best_valid_loss:.4f}')
        print(f'Training time per epoch: {total_train_time/self.epochs:.4f}')

if __name__ == '__main__':
    """
    Accept 5 parameters from command line:
        models: the name of models, including 'gru', 'lstm', and 'bert'.
        dataset: the name of dataset, including 'mba', 'poly1', and 'poly6'.
        train: whether to train the model.
        batch_size: batch size
        epochs: epochs
    """
    parser = argparse.ArgumentParser(description='Execute models.')
    parser.add_argument('--models', type=str, default='gru', help='model: gru, lstm or bert')
    parser.add_argument('--dataset', type=str, default='np_mix', help='dataset, mba, poly1 or poly6')
    parser.add_argument('--train', type=int, default=1, help='whether to train the model')
    parser.add_argument('--batch_size', type=int, default=128, help='size of each mini batch')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')

    args = parser.parse_args()

    dataset = Dataset(args.models, args.dataset)
    train_data, valid_data = dataset.build_dataset()

    print(f'Batch size: {args.batch_size}')
    print(f'Epochs:{args.epochs}')

    input_dim = len(dataset.SRC.vocab)
    output_dim = len(dataset.TGT.vocab)
    src_pad_idx = dataset.SRC.vocab.stoi[dataset.SRC.pad_token]
    tgt_pad_idx = dataset.TGT.vocab.stoi[dataset.TGT.pad_token]
    model = Model(args.models, input_dim, output_dim, src_pad_idx, tgt_pad_idx).build_model()
    model_name = args.models + '_' + args.dataset
    
    if args.train == 1:
        train_iter, valid_iter = BucketIterator.splits(
            (train_data, valid_data),
            batch_size=args.batch_size,
            sort=False,
            device=torch.device('cuda')
        )

        pad_token=dataset.TGT.vocab.stoi[dataset.TGT.pad_token]
        trainer = Trainer(model, args.epochs, pad_token, args.models)
        trainer.run(model, train_iter, valid_iter, model_name)

    model.load_state_dict(torch.load(f'saved_models/{model_name}.pt'))

    verifer = Verifer(dataset.SRC, dataset.TGT, model, args.models)
    verifer.count_acc(valid_data)
