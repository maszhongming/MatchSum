import torch
from torch import nn
from torch.nn import init

from transformers import BertModel, RobertaModel

class MatchSum(nn.Module):
    
    def __init__(self, candidate_num, encoder, hidden_size=768):
        super(MatchSum, self).__init__()
        
        self.hidden_size = hidden_size
        self.candidate_num  = candidate_num
        
        if encoder == 'bert':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.encoder = RobertaModel.from_pretrained('roberta-base')

    def forward(self, text_id, candidate_id, summary_id):
        
        batch_size = text_id.size(0)
        
        # get document embedding
        input_mask = ~(text_id == 0)
        out = self.encoder(text_id, attention_mask=input_mask)[0] # last layer
        doc_emb = out[:, 0, :]
        assert doc_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]
        
        # get summary embedding
        input_mask = ~(summary_id == 0)
        out = self.encoder(summary_id, attention_mask=input_mask)[0] # last layer
        summary_emb = out[:, 0, :]
        assert summary_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]

        # get summary score
        summary_score = torch.cosine_similarity(summary_emb, doc_emb, dim=-1)

        # get candidate embedding
        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = ~(candidate_id == 0)
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num, self.hidden_size)  # [batch_size, candidate_num, hidden_size]
        assert candidate_emb.size() == (batch_size, candidate_num, self.hidden_size)
        
        # get candidate score
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1) # [batch_size, candidate_num]
        assert score.size() == (batch_size, candidate_num)

        return {'score': score, 'summary_score': summary_score}

