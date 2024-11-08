import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Blip.models.blip import blip_feature_extractor


class LinearDropConnect(nn.Linear):  # 继承 nn.Linear  
    def __init__(self, in_features, out_features, bias=True, dropout=0.1):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout
        self._weight = self.weight
        
    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight   # 初始化的权重
        else:
            mask = self.weight.new_empty(   # 同等大小的mask
                self.weight.size(),
                dtype=torch.bool
            )
            mask.bernoulli_(self.dropout)  # 伯努利分布填充mask， 
            self._weight = self.weight.masked_fill(mask, 0.)  # 调整 weight   dropout

    def forward(self, input, sample_mask=False):
        if self.training:  # 训练时才用 dropout
            if sample_mask:
                self.sample_mask()
            return F.linear(input, self._weight, self.bias)
        else:
            return F.linear(input, self.weight * (1 - self.dropout),
                            self.bias)

class MultiHeadAttention(nn.Module):
    def __init__(self,input_size,output_size,d_k,d_v, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.query_linear = nn.Linear(input_size, num_heads*d_k)  # 一头变多头
        self.key_linear = nn.Linear(input_size, num_heads*d_k)
        self.value_linear = nn.Linear(input_size, num_heads*d_v)

        self.final_linear = nn.Linear(num_heads*d_v, output_size)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        matmul_qk = torch.matmul(query, key.transpose(-2, -1)) # num_heads,  batch, mex_len, mex_len

        # 缩放点积注意力
        depth = torch.tensor(key.size(-1), dtype=torch.float32)
        logits = matmul_qk / torch.sqrt(depth)

        # 在mask参数不为None时，将mask应用于注意力权重
        if mask is not None:
            mask = mask.transpose(0,1)
            mask = mask.unsqueeze(0).repeat(matmul_qk.size(0),1,1)
            mask = mask.unsqueeze(-1).repeat(1,1,1,matmul_qk.size(-1))
            
            # print( mask.size(),matmul_qk.size())
            assert mask.size() == matmul_qk.size()  # 大小统一
            logits = logits.masked_fill(mask == 0, float('-inf'))

        # 注意力权重
        attention_weights = nn.functional.softmax(logits, dim=-1)  

        output = torch.matmul(attention_weights, value)  # num_heads,  batch, mex_len, d_v

        return output, attention_weights

    def forward(self, query, key, value, mask=None):
      
        # 分别通过线性变换获取查询、键、值
        query = self.query_linear(query)   # mex_len , batch , num_heads*d_k
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        query = query.view(query.size(0), -1, self.num_heads, self.d_k).transpose(0,2)     # num_heads,  batch, mex_len, d_k      
        key = key.view(key.size(0), -1, self.num_heads, self.d_k).transpose(0,2) 
        value = value.view(value.size(0), -1, self.num_heads, self.d_v).transpose(0,2) 
        
        # 计算注意力权重
        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value, mask=mask)

        # 合并多头
        scaled_attention = scaled_attention.transpose(0, 2).contiguous().view(value.size(2),value.size(1), -1)
        # 经过最后一层线性变换
        output = self.final_linear(scaled_attention)
        return output, attention_weights


class MultiHeadTransformerLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_attention_heads, dropout_rate=0.1):
        super(MultiHeadTransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(input_size,hidden_size,hidden_size,hidden_size, num_attention_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(input_size)

    def forward(self, input_q ,input_kv, sim_score=None):
        # attention
        attention_output, _ = self.attention(input_q, input_kv, input_kv)
        if sim_score != None:
            attention_output = self.layer_norm1(input_kv + torch.mul(sim_score,attention_output))
        else:
            attention_output = self.layer_norm1(input_kv + attention_output)
            
        # Feed-forward layer
        feed_forward_output = self.feed_forward(attention_output)
        output_seq = self.layer_norm2(attention_output + feed_forward_output)

        return output_seq

class ner_unit(nn.Module):
    def __init__(self, args, ner2idx):
        super(ner_unit, self).__init__()
        self.hidden_size =args.hidden_size
        self.ner2idx = ner2idx

        self.hid2hid = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.hid2tag = nn.Linear(self.hidden_size, len(ner2idx))

        self.elu = nn.ELU()  # 激活函数
        self.n = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size)  # 对 -1维度进行当前batch归一化

        self.dropout = nn.Dropout(args.dropout)



    def forward(self, h_ner1, h_ner2, mask,token_pos=None):
        length, batch_size, _ = h_ner2.size()


        h_global = torch.cat((h_ner1, h_ner2), dim=-1)
        h_global = torch.tanh(self.n(h_global))


        #将共享特征于实体特征 全局特征扩充成一个 表，
        h_global = torch.max(h_global, dim=0)[0]  # 一个句子一个特征  batch_size * hidden_size
        h_global = h_global.unsqueeze(0).repeat(h_ner2.size(0), 1, 1)   # max_len*batch_size*hidden_size
        h_global = h_global.unsqueeze(0).repeat(h_ner2.size(0), 1, 1, 1)   # max_len*max_len*batch_size*hidden_size

        st = h_ner2.unsqueeze(1).repeat(1, length, 1, 1)
        en = h_ner2.unsqueeze(0).repeat(length, 1, 1, 1)

        ner = torch.cat((st, en, h_global), dim=-1)
        
        
        ner = self.ln(self.hid2hid(ner))
        ner = self.elu(self.dropout(ner))
        ner = torch.sigmoid(self.hid2tag(ner))

        diagonal_mask = torch.triu(torch.ones(batch_size, length, length)).to(device) # 上三角
        diagonal_mask = diagonal_mask.permute(1, 2, 0)

        mask_s = mask.unsqueeze(1).repeat(1, length, 1)
        mask_e = mask.unsqueeze(0).repeat(length, 1, 1)

        mask_ner = mask_s * mask_e

        if(token_pos!= None):
            mk_s = token_pos.unsqueeze(1).repeat(1, length, 1)
            mk_e = token_pos.unsqueeze(0).repeat(length, 1, 1)
            mask = diagonal_mask * mask_ner *mk_s * mk_e
        else:
            mask = diagonal_mask * mask_ner

        mask = mask.unsqueeze(-1).repeat(1, 1, 1, len(self.ner2idx))

        ner = ner * mask

        return ner

class re_unit(nn.Module):
    def __init__(self, args, re2idx):
        super(re_unit, self).__init__()
        self.hidden_size = args.hidden_size
        self.relation_size = len(re2idx)
        self.re2idx = re2idx

        self.hid2hid = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.hid2rel = nn.Linear(self.hidden_size, self.relation_size)
        self.elu = nn.ELU()

        self.r = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size)

        self.dropout = nn.Dropout(args.dropout)



    def forward(self, h_re1, h_re2, mask,token_pos=None):
        length, batch_size, _ = h_re2.size()

        h_global = torch.cat((h_re1, h_re2), dim=-1)
        h_global = torch.tanh(self.r(h_global))


        h_global = torch.max(h_global, dim=0)[0]
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1)
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1, 1)

        r1 = h_re2.unsqueeze(1).repeat(1, length, 1, 1)
        r2 = h_re2.unsqueeze(0).repeat(length, 1, 1, 1)

        re = torch.cat((r1, r2, h_global), dim=-1)
        
        re = self.ln(self.hid2hid(re))
        re = self.elu(self.dropout(re))
        re = torch.sigmoid(self.hid2rel(re))

        mask = mask.unsqueeze(-1).repeat(1, 1, self.relation_size)
        mask_e1 = mask.unsqueeze(1).repeat(1, length, 1, 1)
        mask_e2 = mask.unsqueeze(0).repeat(length, 1, 1, 1)
        mask = mask_e1 * mask_e2

        re = re * mask

        return re

class RE(nn.Module):
    def __init__(self, args, re2idx):
        super(RE, self).__init__()
        self.hidden_size = args.hidden_size
        self.relation_size = len(re2idx)
        self.re2idx = re2idx

        self.hid2hid = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.hid2rel = nn.Linear(self.hidden_size, self.relation_size)
        self.elu = nn.ELU()

        self.r = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size)

        self.dropout = nn.Dropout(args.dropout)



    def forward(self, h_re1, h_re2, mask,token_pos=None):
        length, batch_size, _ = h_re2.size()

        h_global = torch.cat((h_re1, h_re2), dim=-1)
        h_global = torch.tanh(self.r(h_global))


        h_global = torch.max(h_global, dim=0)[0]
        hre1 = torch.max(h_re1, dim=0)[0]
        hre2 = torch.max(h_re1, dim=0)[0]

        re = torch.cat((hre1,hre2, h_global), dim=-1)
        
        re = self.ln(self.hid2hid(re))
        re = self.elu(self.dropout(re))
        re = self.hid2rel(re)

        return re


class TESGA(nn.Module):
    def __init__(self, args, input_size, ner2idx, rel2idx):
        super(TESGA, self).__init__()
        self.args = args
        
        self.ner = ner_unit(args, ner2idx)
        self.re = re_unit(args, rel2idx)
        self.dropout = nn.Dropout(args.dropout)
        
        self.img_model = blip_feature_extractor(pretrained=args.blip_local_path, image_size=args.img_size, vit='base')
        for param in self.img_model.parameters():  # 冻结预训练模型
            param.requires_grad = False
        
        self.img_freature_transform = LinearDropConnect(input_size, 2*args.hidden_size, bias=True, dropout= args.dropconnect)
        self.text_input_transform = LinearDropConnect(input_size, 3*args.hidden_size, bias=True, dropout= args.dropconnect)
        
        
        self.selftransNER = MultiHeadTransformerLayer(args.hidden_size,args.hidden_size,12)
        self.selftransRE = MultiHeadTransformerLayer(args.hidden_size,args.hidden_size,12)
        
        self.crosstransNER = MultiHeadTransformerLayer(args.hidden_size,args.hidden_size,12)
        self.crosstransRE = MultiHeadTransformerLayer(args.hidden_size,args.hidden_size,12)
        
        self.get_ner_ls_feature = nn.LSTM(args.hidden_size,args.hidden_size,2,bidirectional=False)
        self.get_re_ls_feature = nn.LSTM(args.hidden_size,args.hidden_size,2,bidirectional=False)
        
        
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_local_path,never_split= ["<s>","</s>","<o>","</o>"])
        self.bert = BertModel.from_pretrained(args.bert_local_path)
        


    def forward(self, x, images, sents, sim_score, mask,token_pos = None):

        x = self.tokenizer(x, return_tensors="pt",padding='longest',is_split_into_words=True).to(device)   # 分词
        x = self.bert(**x)[0]  # size  batch size * max_len * 768
        
        
        img_feature = self.img_model(images, sents, mode='multimodal')

        img_feature = torch.tanh(self.img_freature_transform(img_feature))
        imf_1,imf_2=img_feature.chunk(2,-1)
        
        imf_1 = torch.max(imf_1, dim=1)[0]
        imf_2 = torch.max(imf_2, dim=1)[0]
        
        ch_0 = imf_1.unsqueeze(0).repeat(2, 1, 1)  
        # re_ch_0 = imf_1.unsqueeze(0).repeat(2, 1, 1)  
       
        # imf_1 = imf_1.unsqueeze(0).repeat(x.size(1), 1, 1)
        imf_2 = imf_2.unsqueeze(0).repeat(x.size(1), 1, 1)
        
        x = x.transpose(0, 1)  # 转置  max_len * batch size *768
        if self.training:
            x = self.dropout(x)  # dropout tensor
        
        h=self.text_input_transform(x)
        h0,h1,h2=h.chunk(3,-1)
        
        h_ner,_ = self.get_ner_ls_feature(h0,(ch_0,ch_0))
        h_re,_ = self.get_re_ls_feature(h0,(ch_0,ch_0))
        
        h_ner2 = self.crosstransNER(imf_2,self.selftransNER(h_ner,h1),sim_score)
        h_re2 = self.crosstransRE(imf_2,self.selftransRE(h_re,h2),sim_score)
        
        if(token_pos!= None):
            token_pos = token_pos.transpose(0,1)
        
        ner_score = self.ner(h_ner, h_ner2, mask,token_pos)
        re_core = self.re(h_re, h_re2, mask,token_pos)
        return ner_score, re_core
    


class MCRE(nn.Module):
    def __init__(self, args, input_size, ner2idx, rel2idx):
        super(MCRE, self).__init__()
        self.args = args

        self.re = RE(args, rel2idx)
        self.dropout = nn.Dropout(args.dropout)
        
        self.img_model = blip_feature_extractor(pretrained=args.blip_local_path, image_size=args.img_size, vit='base')
        for param in self.img_model.parameters():  # 冻结预训练模型
            param.requires_grad = False
        
        self.img_freature_transform = LinearDropConnect(input_size, 2*args.hidden_size, bias=True, dropout= args.dropconnect)
        self.text_input_transform = LinearDropConnect(input_size, 2*args.hidden_size, bias=True, dropout= args.dropconnect)
        
        self.selftransRE = MultiHeadTransformerLayer(args.hidden_size,args.hidden_size,12)
        
        self.crosstransRE = MultiHeadTransformerLayer(args.hidden_size,args.hidden_size,12)
        self.get_re_ls_feature = nn.LSTM(args.hidden_size,args.hidden_size,2,bidirectional=False)
        
        
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_local_path,never_split= ["<s>","</s>","<o>","</o>"])
        self.bert = BertModel.from_pretrained(args.bert_local_path)
        


    def forward(self, x, images, sents, sim_score, mask,token_pos = None):

        x = self.tokenizer(x, return_tensors="pt",padding='longest',is_split_into_words=True).to(device)   # 分词
        x = self.bert(**x)[0]  # size  batch size * max_len * 768
        
        
        img_feature = self.img_model(images, sents, mode='multimodal')
        
        img_feature = torch.tanh(self.img_freature_transform(img_feature))
        imf_1,imf_2=img_feature.chunk(2,-1)
        
        imf_1 = torch.max(imf_1, dim=1)[0]
        imf_2 = torch.max(imf_2, dim=1)[0]
        
        re_ch_0 = imf_1.unsqueeze(0).repeat(2, 1, 1)  
       
        imf_2 = imf_2.unsqueeze(0).repeat(x.size(1), 1, 1)

        
        x = x.transpose(0, 1)  # 转置  max_len * batch size *768
        if self.training:
            x = self.dropout(x)  # dropout tensor
        
        h=self.text_input_transform(x)
        h1,h2=h.chunk(2,-1)
        
        
        h_re1,_ = self.get_re_ls_feature(h1,(re_ch_0,re_ch_0))
        
        h_re2 = self.crosstransRE(imf_2,self.selftransRE(h_re1,h2),sim_score)
        
        if(token_pos!= None):
            token_pos = token_pos.transpose(0,1)
        
        re_core = self.re(h_re1, h_re2, mask,token_pos)
        return None, re_core



class MCNER(nn.Module):
    def __init__(self, args, input_size, ner2idx, rel2idx):
        super(MCNER, self).__init__()
        self.args = args
        
        self.ner = ner_unit(args, ner2idx)
        self.dropout = nn.Dropout(args.dropout)
        
        self.img_model = blip_feature_extractor(pretrained=args.blip_local_path, image_size=args.img_size, vit='base')
        for param in self.img_model.parameters():  # 冻结预训练模型
            param.requires_grad = False
        
        self.img_freature_transform = LinearDropConnect(input_size, 2*args.hidden_size, bias=True, dropout= args.dropconnect)
        self.text_input_transform = LinearDropConnect(input_size, 2*args.hidden_size, bias=True, dropout= args.dropconnect)
        
        self.selftransNER = MultiHeadTransformerLayer(args.hidden_size,args.hidden_size,12)
        
        self.crosstransNER = MultiHeadTransformerLayer(args.hidden_size,args.hidden_size,12)
        
        self.get_ls_feature = nn.LSTM(args.hidden_size,args.hidden_size,2,bidirectional=False)
        self.get_re_ls_feature = nn.LSTM(args.hidden_size,args.hidden_size,2,bidirectional=False)
        
        
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_local_path,never_split= ["<s>","</s>","<o>","</o>"])
        self.bert = BertModel.from_pretrained(args.bert_local_path)
        


    def forward(self, x, images, sents, sim_score, mask,token_pos = None):

        x = self.tokenizer(x, return_tensors="pt",padding='longest',is_split_into_words=True).to(device)   # 分词
        x = self.bert(**x)[0]  # size  batch size * max_len * 768
        
        img_feature = self.img_model(images, sents, mode='multimodal')

        img_feature = torch.tanh(self.img_freature_transform(img_feature))
        imf_1,imf_2=img_feature.chunk(2,-1)
        
        imf_1 = torch.max(imf_1, dim=1)[0]
        imf_2 = torch.max(imf_2, dim=1)[0]
        
        ner_ch_0 = imf_1.unsqueeze(0).repeat(2, 1, 1)  
        # re_ch_0 = imf_1.unsqueeze(0).repeat(2, 1, 1)  
       
        # imf_1 = imf_1.unsqueeze(0).repeat(x.size(1), 1, 1)
        imf_2 = imf_2.unsqueeze(0).repeat(x.size(1), 1, 1)
        
        x = x.transpose(0, 1)  # 转置  max_len * batch size *768
        if self.training:
            x = self.dropout(x)  # dropout tensor
        
        h=self.text_input_transform(x)
        h1,h2=h.chunk(2,-1)
        
        h_ls,_ = self.get_ls_feature(h1,(ner_ch_0,ner_ch_0))
        # h_re1,_ = self.get_re_ls_feature(h2,((re_ch_0,re_ch_0)))
        
        h_ner2 = self.crosstransNER(imf_2,self.selftransNER(h_ls,h2),sim_score)
        
        if(token_pos!= None):
            token_pos = token_pos.transpose(0,1)
        
        ner_score = self.ner(h_ls, h_ner2, mask,token_pos)
        
        return ner_score, None