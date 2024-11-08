import json
import os
import torch

def json_load(path, name):
    file = path + "/" + name
    with open(file,'r') as f:
        return json.load(f)

def json_loads(path, name):
    file = path + "/" + name
    data = []
    with open(file,'r') as f:
        for line in f:
            a = json.loads(line)
            data.append(a)
    return data

def gen_ner_labels(ner_list,l, ner2idx):
    labels = torch.FloatTensor(l,l,len(ner2idx)).fill_(0)
    for i in range(0,len(ner_list),3):
        head = ner_list[i]
        tail = ner_list[i+1]
        n = ner2idx[ner_list[i+2]]
        labels[head][tail][n] = 1

    return labels


def gen_rc_labels(rc_list, l, rel2idx):
    labels = torch.FloatTensor(l, l, len(rel2idx)).fill_(0)
    for i in range(0, len(rc_list), 3):
        e1 = rc_list[i]    # 头节点的索引
        e2 = rc_list[i + 1]   # 尾节点的索引
        r = rc_list[i + 2]  # 关系标签  
        labels[e1][e2][rel2idx[r]] = 1   # 关系

    return labels

def gen_r_labels(r, rel2idx):
    labels = torch.FloatTensor(len(rel2idx)).fill_(0)
    labels[rel2idx[r]]=1

    return labels


def mask_to_tensor(len_list, batch_size):
    token_len = max(len_list)
    tokens = torch.LongTensor(token_len, batch_size).fill_(0)
    for i, s in enumerate(len_list):
        tokens[:s, i] = 1

    return tokens

def sent_to_tensor(len_list, batch_size, sent_list):
    token_len = max(len_list)
    tokens = torch.LongTensor(token_len, batch_size).fill_(0)
    for i, s in enumerate(len_list):
        tokens[:s, i] = torch.tensor(sent_list[i], dtype=torch.long)

    return tokens

class collater():
    def __init__(self, ner2idx, rel2idx,ifpos=False):
        self.ner2idx = ner2idx
        self.rel2idx = rel2idx
        self.ifpos = ifpos

    def __call__(self, data):
        words = [item[0] for item in data]
        ner_labels = [item[1] for item in data]
        rc_labels = [item[2] for item in data]  # 关系 
        bert_len = [item[3] for item in data]  
        sim_score = [item[4] for item in data]   
        images = torch.cat([item[5] for item in data],0)
        sents = [item[6] for item in data]
        
        
        batch_size = len(words)
        s_o_indes =None

        if (ner_labels[0] != None) and (rc_labels[0] != None):
            max_len = max(bert_len)
            ner_labels = [gen_ner_labels(ners, max_len, self.ner2idx) for ners in ner_labels]  # 获得实体标签邻接矩阵
            rc_labels = [gen_rc_labels(rcs,max_len, self.rel2idx) for rcs in rc_labels]   # 获得关系标签邻接矩阵

            ner_labels = torch.stack(ner_labels, dim=2)   # l*l*batchsize*len_n  
            rc_labels = torch.stack(rc_labels, dim=2)   # l*l*batchsize*len_r
            
        elif ner_labels[0] != None:
            max_len = max(bert_len)
            ner_labels = [gen_ner_labels(ners, max_len, self.ner2idx) for ners in ner_labels]  # 获得实体标签邻接矩阵
    
            ner_labels = torch.stack(ner_labels, dim=2)   # l*l*batchsize*len_n  
            rc_labels = rc_labels   #[None,None]
            
        else:
            # max_len = max(bert_len)
            # ner_labels = ner_labels
            # rc_labels = [gen_rc_labels(rcs,max_len, self.rel2idx) for rcs in rc_labels]   # 获得关系标签邻接矩阵
            # rc_labels = torch.stack(rc_labels, dim=2)   # l*l*batchsize*len_r
            
            ner_labels = ner_labels
            s_o_indes = [rcs[:2] for rcs in rc_labels]
            rc_labels = [self.rel2idx[rcs[-1]] for rcs in rc_labels]
            rc_labels =  torch.tensor(rc_labels, dtype=torch.long)
            

            
            
        mask = mask_to_tensor(bert_len, batch_size)  # 给句子打上mask

        if(self.ifpos):
            token_pos = [item[5] for item in data]
            token_len = max(bert_len)
            tokens = torch.LongTensor(batch_size,token_len).fill_(0)
            for i, s in enumerate(bert_len):
                tokens[i, 1:s-1] = torch.LongTensor(token_pos[i])

            return [words,sim_score,ner_labels,rc_labels,mask,images,sents,tokens]
        else:
            return [words,sim_score,ner_labels,rc_labels,mask,images,sents,s_o_indes]

    
class save_results(object):
    def __init__(self, filename, header=None):
        self.filename = filename
        if os.path.exists(filename):
            os.remove(filename)

        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)

    def save(self, info):
        with open(self.filename, 'a') as out:
            print(info, file=out)


def map_to_idx(text,word2idx):
    ids = [word2idx[t] if t in word2idx else 1 for t in text]
    return ids

def is_overlap(ent1, ent2):
    if ent1[1] < ent2[0] or ent2[1] < ent1[0]:
        return False
    return True

