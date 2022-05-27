# from transformers import (ElectraModel, ElectraPreTrainedModel, BertModel, BertPreTrainedModel, RobertaModel)
# from _typeshed import Self
from transformers import (ElectraModel, ElectraPreTrainedModel, BertModel, BertPreTrainedModel, RobertaModel)
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, Conv1d
import torch.nn.functional as F
import math
import torch.nn.utils.rnn as rnn_utils

BertLayerNorm = torch.nn.LayerNorm

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (batch_size, seq_len, all_head_size) -> (batch_size, num_attention_heads, seq_len, attention_head_size)

    def forward(self, input_ids_a, input_ids_b, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(input_ids_a)
        mixed_key_layer = self.key(input_ids_b)
        mixed_value_layer = self.value(input_ids_b)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # (batch_size * num_choice, num_attention_heads, seq_len, attention_head_size) -> (batch_size * num_choice, seq_len, num_attention_heads, attention_head_size)

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        
        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids_a + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)

class MyLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, g_sz):
        super(MyLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.g_sz = g_sz
        self.all1 = nn.Linear((self.hidden_sz * 1 + self.input_sz  * 1),  self.hidden_sz)
        self.all2 = nn.Linear((self.hidden_sz * 1 + self.input_sz  +self.g_sz), self.hidden_sz)
        self.all3 = nn.Linear((self.hidden_sz * 1 + self.input_sz  +self.g_sz), self.hidden_sz)
        self.all4 = nn.Linear((self.hidden_sz * 1 + self.input_sz  * 1), self.hidden_sz)

        self.all11 = nn.Linear((self.hidden_sz * 1 + self.g_sz),  self.hidden_sz)
        self.all44 = nn.Linear((self.hidden_sz * 1 + self.g_sz), self.hidden_sz)

        self.init_weights()
        self.drop = nn.Dropout(0.5)
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def node_forward(self, xt, ht, Ct_x, mt, Ct_m):

        # # # new standard lstm
        hx_concat = torch.cat((ht, xt), dim=1)
        hm_concat = torch.cat((ht, mt), dim=1)
        hxm_concat = torch.cat((ht, xt, mt), dim=1)


        i = self.all1(hx_concat)
        o = self.all2(hxm_concat)
        f = self.all3(hxm_concat)
        u = self.all4(hx_concat)
        ii = self.all11(hm_concat)
        uu = self.all44(hm_concat)

        i, f, o, u = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(u)
        ii,uu = torch.sigmoid(ii), torch.tanh(uu)
        Ct_x = i * u + ii * uu + f * Ct_x
        ht = o * torch.tanh(Ct_x) 

        return ht, Ct_x, Ct_m 

    def forward(self, x, m, init_stat=None):
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        cell_seq = []
        if init_stat is None:
            ht = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_x = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_m = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
        else:
            ht, Ct = init_stat
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            mt = m[:, t, :]
            ht, Ct_x, Ct_m= self.node_forward(xt, ht, Ct_x, mt, Ct_m)
            hidden_seq.append(ht)
            cell_seq.append(Ct_x)
            if t == 0:
                mht = ht
                mct = Ct_x
            else:
                mht = torch.max(torch.stack(hidden_seq), dim=0)[0]
                mct = torch.max(torch.stack(cell_seq), dim=0)[0]
        hidden_seq = torch.stack(hidden_seq).permute(1, 0, 2) ##batch_size x max_len x hidden
        return hidden_seq

class Bert_v4(BertPreTrainedModel):
    def __init__(self, config, lstm_hidden_size=128, lstm_num_layers=1, gcn_layer=1, mylstm_hidden_size=128, num_decoupling=1):
        super().__init__(config)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.num_decoupling = num_decoupling
        # self.graph_dim = graph_dim
        self.gcn_layer = gcn_layer
        self.mylstm_hidden_size = mylstm_hidden_size

        self.bert = BertModel(config)
        self.BiLSTM = nn.LSTM(2*mylstm_hidden_size, self.lstm_hidden_size, self.lstm_num_layers, bias=True, batch_first=True, bidirectional=True)
        self.graph_dim = config.hidden_size
        self.lstm_f = MyLSTM(config.hidden_size, self.mylstm_hidden_size, self.graph_dim) 
        self.lstm_b = MyLSTM(config.hidden_size, self.mylstm_hidden_size, self.graph_dim)
        self.drop_lstm = nn.Dropout(config.hidden_dropout_prob)

        self.pooler = nn.Linear(8*self.mylstm_hidden_size, 4*self.mylstm_hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(4*self.mylstm_hidden_size, 1)
        # self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.SASelfMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.W = nn.ModuleList()
        for layer in range(self.gcn_layer):
            self.W.append(nn.Linear(self.graph_dim, self.graph_dim))

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        sep_pos = None,
        position_ids=None,
        turn_ids = None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        adj_matrix_speaker = None,
        adj_matrix_mention = None
    ):
        adj_matrix = adj_matrix_mention.float()
        # (batch_size, choice, seq_len)
        num_labels = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None 
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        orig_attention_mask = attention_mask
        # (batch_size * choice, seq_len)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # sep_pos = sep_pos.view(-1, sep_pos.size(-1)) if sep_pos is not None else None
        # turn_ids = turn_ids.view(-1, turn_ids.size(-1)) if turn_ids is not None else None
        
        # turn_ids = turn_ids.unsqueeze(-1).repeat([1,1,turn_ids.size(1)])
        
        #print("sep_pos:", sep_pos)
        
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        
        # #print("size of sequence_output:", sequence_output.size())
        adj_matrix_speaker = adj_matrix_speaker.unsqueeze(1)
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        # attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        # attention_mask = (1.0 - attention_mask) * -10000.0

        outputs = self.bert(
            input_ids,
            attention_mask=orig_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0] # (batch_size * num_choice, seq_len, hidden_size)
        cls_rep = sequence_output[:,0,:] #(batch_size*num_chioce, hidden_size)
        hidden_size = sequence_output.size(-1)
        cls_rep = cls_rep.view(-1, num_labels, hidden_size) #(batch_size, num_chioce, hidden_size)

        adj_matrix_speaker = adj_matrix_speaker.unsqueeze(1)
        sa_self_mask = (1.0 - adj_matrix_speaker) * -10000.0
        sa_self_ = self.SASelfMHA[0](cls_rep, cls_rep, attention_mask = sa_self_mask)[0]
        for t in range(1, self.num_decoupling):
            sa_self_ = self.SASelfMHA[t](sa_self_, sa_self_, attention_mask = sa_self_mask)[0]
        with_sa_self = self.linear(torch.cat((cls_rep,sa_self_),2))#(batch_size, num_chioce, hidden_size)

        # orig_lstm_output, (hn, cn) = self.BiLSTM(cls_rep) #(batch_size, num_chioce, 2*lstm_hidden_size)
        # target = orig_lstm_output[:,0,:] #(batch_size, 2*lstm_hidden_size)
        # lstm_output = orig_lstm_output.repeat(1,1,4) #(batch_size, num_chioce, 2*lstm_hidden_size *4)
        # for i in lstm_output.shape(0):
        #     for j in lstm_output.shape(1):
        #         a = lstm_output[i,j,:]
        #         b = target[i,:]
        #         c = a * b
        #         d = a - b
        #         lstm_output[i,j,:] = torch.cat((a,b,c,d)) #(2*lstm_hidden_size *4)
        # adj_matrix = adj_matrix.to(self.device) #(batch_size, num_choice, num_choice)
        batch_size, sent_len, input_dim = cls_rep.size()#(batch_size, num_chioce, 2*lstm_hidden_size)
        #adj_matrix (batch_size, num_choice, num_choice)
        denom = adj_matrix.sum(2).unsqueeze(2) + 1 #d

        graph_input = with_sa_self[:, :, :self.graph_dim]#(batch_size, num_chioce, 2*lstm_hidden_size)
        # print('graph-type:', graph_input.dtype)
        # print('adj_type:', adj_matrix.dtype)
        # print('lstmoutput:', orig_lstm_output.dtype)
        # print('sequenceoutput:', sequence_output.dtype)

        for l in range(self.gcn_layer):
            Ax = adj_matrix.bmm(graph_input)  ## N x N  times N x h  = Nxh #(batch_size, num_chioce, 2*lstm_hidden_size)
            AxW = self.W[l](Ax)   ## N x m #(batch_size, num_chioce, 2*lstm_hidden_size)
            AxW = AxW + self.W[l](graph_input)  ## self loop  N x h #(batch_size, num_chioce, 2*lstm_hidden_size)
            AxW = AxW / denom
            graph_input = torch.relu(AxW)#(batch_size, num_chioce, 2*lstm_hidden_size)
        # forward LSTM
        lstm_out_f = self.lstm_f(with_sa_self, graph_input)#(batch_size, num_chioce, mylstm_hidden_size)
        # backward LSTM
        # word_rep_b = masked_flip(cls_rep, word_seq_len.tolist())
        # c_b = masked_flip(graph_input, word_seq_len.tolist())
        # cls_rep_b = torch.flip(cls_rep, [1])
        with_sa_self_b = torch.flip(with_sa_self, [1])
        graph_input_b = torch.flip(graph_input, [1])
        lstm_out_b = self.lstm_b(with_sa_self_b, graph_input_b)#(batch_size, num_chioce, mylstm_hidden_size)
        lstm_out_b = torch.flip(lstm_out_b, [1])

        lstm_output = torch.cat((lstm_out_f, lstm_out_b), dim=2)#(batch_size, num_chioce, 2*mylstm_hidden_size)
        lstm_output, (hn, cn) = self.BiLSTM(lstm_output) #(batch_size, num_chioce, 2*lstm_hidden_size)

        lstm_output = self.drop_lstm(lstm_output)

        target = lstm_output[:,0,:] #(batch_size, 2*mylstm_hidden_size)
        final_lstm_output = lstm_output.repeat(1,1,4) #(batch_size, num_chioce, 2*mylstm_hidden_size *4)
        for i in range(final_lstm_output.size(0)):
            for j in range(final_lstm_output.size(1)):
                a = lstm_output[i,j,:]
                b = target[i,:]
                c = a * b
                d = a - b
                final_lstm_output[i,j,:] = torch.cat((a,b,c,d)) #(2*mylstm_hidden_size *4)

        pooled_output = self.pooler_activation(self.pooler(final_lstm_output)) #(batch_size, num_chioce, 4*mylstm_hidden_size )
        pooled_output = self.dropout(pooled_output)
        
        # if num_labels > 2:
        logits = self.classifier(pooled_output) #(batch_size, num_chioce, 1)
        reshaped_logits = logits.squeeze(2) #(batch_size, num_chioce)
        # else:
        #     logits = self.classifier2(pooled_output)

        # reshaped_logits = logits.view(-1, num_labels) if num_labels > 2 else logits #(batch_size,num_chioce)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs #(loss), reshaped_logits, (hidden_states), (attentions)
    