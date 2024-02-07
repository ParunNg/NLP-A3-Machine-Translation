import torch
import math
from torch import nn
import torch.nn.functional as F

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, trg_eos_idx, device):
        super().__init__()
        # store the input parameters so we can retrive them later for model inferencing
        self.params = {'encoder': encoder, 'decoder': decoder,
                       'src_pad_idx': src_pad_idx,
                       'trg_pad_idx': trg_pad_idx, 'trg_eos_idx': trg_eos_idx}
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_eos_idx = trg_eos_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention
    
    def generate(self, src, max_len=100):

        self.eval()

        with torch.no_grad():
            src_mask = self.make_src_mask(src)
            enc_src = self.encoder(src, src_mask)

            trg = torch.ones((src.shape[0], 1), device=self.device, dtype=torch.long) * self.trg_pad_idx

            for i in range(1, max_len):
                trg_mask = self.make_trg_mask(trg)
                output, _ = self.decoder(trg, enc_src, trg_mask, src_mask)
                pred_token = output.argmax(2)[:, -1].unsqueeze(1)
                trg = torch.cat((trg, pred_token), dim=1)

                if pred_token.item() == self.trg_eos_idx:
                    break

            return trg[:, 1:]