import json
from utils.helper import *
from transformers import BertTokenizer
from torch.utils.data import Dataset,DataLoader
import random

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def load_demo_image(image_size,img_path):
    
    raw_image = Image.open(img_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0)
    return image

class dataprocess(Dataset):
    def __init__(self, data, sim_mode,img_path,bert_local_path,img_size = 384,ifpos = False,never_split_tokens=None,dataset="JMERE"):
        self.data = data
        self.ifpos = ifpos
        self.sim_mode = sim_mode
        self.img_path = img_path
        self.img_size = img_size
        self.dataset = dataset
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_local_path,never_split=never_split_tokens)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if(self.ifpos):
            words = self.data[idx][0]
            itm_score = self.data[idx][1]
            itc_score = self.data[idx][2]
            token_pos = self.data[idx][3]
            ner_labels = self.data[idx][4]
            rc_labels = self.data[idx][5]
            img_name = self.data[idx][6]
        else:
            words = self.data[idx][0]
            itm_score = self.data[idx][1]
            itc_score = self.data[idx][2]
            ner_labels = self.data[idx][3]
            rc_labels = self.data[idx][4]
            img_name = self.data[idx][5]
        
        sent_str = ' '.join(words)
        bert_words = self.tokenizer.tokenize(sent_str)
        bert_len = len(bert_words) + 2
        # bert_len = original sentence + [CLS] and [SEP]

        if self.dataset == "JMERE":
            word_to_bep = self.map_origin_word_to_bert(words)
            ner_labels = self.ner_label_transform(ner_labels, word_to_bep)   # 将词位置转换成了  bert分词后的位置
            rc_labels = self.rc_label_transform(rc_labels, word_to_bep)  # r同理
            
        elif self.dataset == "MNRE":
            word_to_bep = self.map_origin_word_to_bert(words)
            ner_labels = None
            rc_labels = self.rc_label_transform(rc_labels, word_to_bep)  # r同理

        elif self.dataset == "twitter17" or (self.dataset == "twitter15"):
            word_to_bep = self.map_origin_word_to_bert(words)
            ner_labels = self.ner_label_transform(ner_labels, word_to_bep)   # 将词位置转换成了  bert分词后的位置
            rc_labels = None  
            
        # 读取image
        image = load_demo_image(self.img_size,self.img_path+img_name)  # 3* img_size * img_size
        
        if self.sim_mode == "itc":
            sim_score = itc_score 
        elif self.sim_mode == "itm":
            sim_score = itm_score 
        else:
            print("error")
        
        if(self.ifpos):
            return (words, ner_labels, rc_labels, bert_len,sim_score,image,sent_str,token_pos)
        else:
            return (words, ner_labels, rc_labels, bert_len,sim_score,image,sent_str)

    def map_origin_word_to_bert(self, words):
        bep_dict = {}
        current_idx = 0
        for word_idx, word in enumerate(words):
            bert_word = self.tokenizer.tokenize(word)
            word_len = len(bert_word)
            bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
            current_idx = current_idx + word_len
        return bep_dict

    def ner_label_transform(self, ner_label, word_to_bert):
        new_ner_labels = []

        for i in range(0, len(ner_label), 3):
            # +1 for [CLS]
            sta = word_to_bert[ner_label[i]][0] + 1
            end = word_to_bert[ner_label[i + 1]][0] + 1
            new_ner_labels += [sta, end, ner_label[i + 2]]

        return new_ner_labels

    def rc_label_transform(self, rc_label, word_to_bert):
        new_rc_labels = []

        for i in range(0, len(rc_label), 3):
            # +1 for [CLS]
            e1 = word_to_bert[rc_label[i]][0] + 1
            e2 = word_to_bert[rc_label[i + 1]][0] + 1
            new_rc_labels += [e1, e2, rc_label[i + 2]]

        return new_rc_labels
    
    

def jmere_preprocess(data,ifpos = False):
    processed = []
    for dic in data:
        text = dic['text']
        text = text.split(" ")
        ner_labels = []
        rc_labels = []
        trips = dic['triple_list']
        img_name = dic["img_id"]
        
        itm_score = dic["itm_score"]
        itc_score = dic["itc_score"]
        
        if "ent_pair_list" in dic.keys():
            ent_pairs =dic["ent_pair_list"]
            for index ,trip in enumerate(trips):
                subj = ent_pairs[index][0][0]
                subj1 = ent_pairs[index][0][1]-1
                obj = ent_pairs[index][1][0]
                obj1 = ent_pairs[index][1][1]-1
                
                rel = trip[1]
                if subj not in ner_labels:
                    # ner_labels +=[subj,subj1,"None"]
                    ner_labels +=[subj,subj1,trip[3]]
                if obj not in ner_labels:
                    # ner_labels +=[obj,obj1,"None"]
                    ner_labels +=[obj,obj1,trip[4]]
                rc_labels+=[subj,obj,rel]
                
        else:  
            for trip in trips:
                subj = text.index(trip[0])
                obj = text.index(trip[2])
                rel = trip[1]
                if subj not in ner_labels:
                    ner_labels +=[subj,subj,"None"]
                if obj not in ner_labels:
                    ner_labels +=[obj,obj,"None"]
                rc_labels+=[subj,obj,rel]
        
        if(ifpos):
            token_pos = dic["token_pos"]
            processed += [(text,itm_score,itc_score,token_pos,ner_labels,rc_labels,img_name)]
        else:
            processed += [(text,itm_score,itc_score,ner_labels,rc_labels,img_name)]

    return processed

def twit_preprocess(data,ifpos = False):
    processed = []
    for dic in data:
        text = dic['text']
        text = text.split(" ")
        ner_labels = []
        img_name = dic["img_id"]
        
        itm_score = dic["itm_score"]
        itc_score = dic["itc_score"]
        
        
        ents =dic["ents"]
        for ent in ents:
            subj = ent["pos"][0]
            subj1 = ent["pos"][1]-1
            if subj not in ner_labels:
                ner_labels +=[subj,subj1,ent['tag']]
            
        if(ifpos):
            token_pos = dic["token_pos"]
            processed += [(text,itm_score,itc_score,token_pos,ner_labels,None,img_name)]
        else:
            processed += [(text,itm_score,itc_score,ner_labels,None,img_name)]

    return processed

def mnre_preprocess(data, ifpos = False):
    processed = []
    for dic in data:
        text = dic['text']
        text = text.split(" ")
        ner_labels = []
        rc_labels = []
        trips = dic['triple_list']
        img_name = dic["img_id"]
        ent_pairs =dic["ent_pair_list"]
        itm_score = dic["itm_score"]
        itc_score = dic["itc_score"]
        
        text.insert(ent_pairs[0][0],"<s>")
        text.insert(ent_pairs[0][1]+1,"</s>")
        subj = ent_pairs[0][0]
        subj1 = ent_pairs[0][1]+1
        
        idx1 = ent_pairs[1][0]
        idx2 = ent_pairs[1][1]+1
        
        if ent_pairs[1][0] > ent_pairs[0][1]:
            idx1 +=2
        elif ent_pairs[1][0] > ent_pairs[0][0]:
            idx1 +=1
            subj1 +=1 
        else:
            subj +=1 
            subj1 +=1 
        
        if ent_pairs[1][1] > ent_pairs[0][1]:
            idx2 +=2
        elif ent_pairs[1][1] > ent_pairs[0][0]:
            subj1 +=1
            idx2 +=1
        else:
            subj +=1
            subj1 +=1
        
        text.insert(idx1,'<o>')
        text.insert(idx2,'</o>')
            
        
        obj = idx1
        obj1 = idx2
        
        rel = trips[2]
        if subj not in ner_labels:
            ner_labels +=[subj,subj1,"None"]
            # ner_labels +=[subj,subj1,trip[3]]
        if obj not in ner_labels:
            ner_labels +=[obj,obj1,"None"]
            # ner_labels +=[obj,obj1,trip[4]]
        rc_labels+=[subj,obj,rel]
            
                

        
        if(ifpos):
            token_pos = dic["token_pos"]
            processed += [(text,itm_score,itc_score,token_pos,ner_labels,rc_labels,img_name)]
        else:
            processed += [(text,itm_score,itc_score,ner_labels,rc_labels,img_name)]

    return processed


def dataloader(args, ner2idx, rel2idx):
    custom_vocab = ["<s>","</s>","<o>","</o>"]
    
    if args.data == "JMERE":
        path = "datasets/" + args.data +"/JMERE_new2"
        img_path = "datasets/" + args.data +"/JMERE_imgs/"
        train_data = json_load(path, 'new_train_triples.json')
        test_data = json_load(path, 'new_test_triples.json')
        dev_data = json_load(path, 'new_val_triples.json')
        
    elif args.data =="MNRE":
        path = "datasets/" + args.data +"/mnre_txt_new2"
        img_path = "datasets/" + args.data +"/mnre_image/img_org/" 
        train_data = json_load(path, 'new_train_triples.json')
        test_data = json_load(path, 'new_test_triples.json')
        dev_data = json_load(path, 'new_val_triples.json')
        
    elif args.data =="twitter17":
        path = "datasets/" + args.data +"/twitter17_new2"
        img_path = "datasets/" + args.data +"/twitter2017_images/" 
        train_data = json_load(path, 'new_train.json')
        test_data = json_load(path, 'new_test.json')
        dev_data = json_load(path, 'new_valid.json')
    
    elif args.data =="twitter15":
        path = "datasets/" + args.data +"/twitter15_new2"
        img_path = "datasets/" + args.data +"/twitter2015_images/" 
        train_data = json_load(path, 'new_train.json')
        test_data = json_load(path, 'new_test.json')
        dev_data = json_load(path, 'new_valid.json')
            
    else:
        print("not find the datasets")
        pass

    if args.data=="JMERE":        
        train_data = jmere_preprocess(train_data,ifpos=False)
        test_data = jmere_preprocess(test_data,ifpos=False)
        dev_data = jmere_preprocess(dev_data,ifpos=False)
        
    elif args.data=="MNRE":        
        train_data = mnre_preprocess(train_data,ifpos=False)
        test_data = mnre_preprocess(test_data,ifpos=False)
        dev_data = mnre_preprocess(dev_data,ifpos=False)
    
    elif args.data=="twitter17" or (args.data == "twitter15"):        
        train_data = twit_preprocess(train_data,ifpos=False)
        test_data = twit_preprocess(test_data,ifpos=False)
        dev_data = twit_preprocess(dev_data,ifpos=False)
        

    if args.data=="twitter17" or (args.data == "twitter15"): 
        train_dataset = dataprocess(train_data, args.sim_mode, img_path, args.bert_local_path, args.img_size, ifpos=False, never_split_tokens=custom_vocab,dataset=args.data)
        test_dataset = dataprocess(test_data, args.sim_mode, img_path,args.bert_local_path, args.img_size, ifpos=False, never_split_tokens=custom_vocab,dataset=args.data)
        dev_dataset = dataprocess(dev_data, args.sim_mode, img_path, args.bert_local_path, args.img_size, ifpos=False, never_split_tokens=custom_vocab,dataset=args.data)

    else:
        train_dataset = dataprocess(train_data, args.sim_mode, img_path+"train/", args.bert_local_path, args.img_size, ifpos=False, never_split_tokens=custom_vocab,dataset=args.data)
        test_dataset = dataprocess(test_data, args.sim_mode, img_path+"test/",args.bert_local_path, args.img_size, ifpos=False, never_split_tokens=custom_vocab,dataset=args.data)
        dev_dataset = dataprocess(dev_data, args.sim_mode, img_path+"val/", args.bert_local_path, args.img_size, ifpos=False, never_split_tokens=custom_vocab,dataset=args.data)
    collate_fn = collater(ner2idx, rel2idx,ifpos=False) # 


    train_batch = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    test_batch = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)
    dev_batch = DataLoader(dataset=dev_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)


    return train_batch, test_batch, dev_batch
