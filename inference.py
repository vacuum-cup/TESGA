import random
import json
import logging
import sys
import torch
import os
import argparse
import numpy as np
from utils.metrics import *
from utils.helper import *
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from model.Multichan_BLIP_retrieval_final import TESGA
from transformers import BertTokenizer
import torch.optim as optim
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S',  # 日期格式
                    level=logging.INFO)   # 级别
logger = logging.getLogger(__name__)   # 日志处理器

def evaluate(test_batch, rel2idx, ner2idx, args, test_or_dev):
    steps, test_loss = 0, 0
    total_triple_num = [0, 0, 0]
    total_entity_num = [0, 0, 0]
    if args.eval_metric == "macro":
        total_triple_num *= len(rel2idx)
        total_entity_num *= len(ner2idx)

    if args.eval_metric == "micro":
        metric = micro(rel2idx, ner2idx)
    else:
        metric = macro(rel2idx, ner2idx)

    with torch.no_grad():
        for data in test_batch:
            steps += 1
            
            text = data[0]  
            sim_scores = torch.Tensor(data[1]).unsqueeze(1).to(device)
            ner_label = data[2].to(device)   # 实体标签
            re_label = data[3].to(device)   # 关系标签
            mask = data[4].to(device) 
            images = data[5].to(device)  
            sents = data[6] 
            # token_pos = data[7].to(device)
            
            token_pos=None
            ner_pred, re_pred = model(text, images, sents,sim_scores, mask,token_pos)
            

            loss = BCEloss(ner_pred, ner_label, re_pred, re_label)
            test_loss += loss

            entity_num = metric.count_ner_num(ner_pred, ner_label)
            triple_num = metric.count_num(ner_pred, ner_label, re_pred, re_label)

            for i in range(len(entity_num)):
                total_entity_num[i] += entity_num[i]
            for i in range(len(triple_num)):
                total_triple_num[i] += triple_num[i]

        triple_result = f1(total_triple_num)
        entity_result = f1(total_entity_num)

        logger.info("------ {} Results ------".format(test_or_dev))
        logger.info("loss : {:.4f}".format(test_loss / steps))
        logger.info("entity: p={:.4f}, r={:.4f}, f={:.4f}".format(entity_result["p"], entity_result["r"], entity_result["f"]))
        logger.info("triple: p={:.4f}, r={:.4f}, f={:.4f}".format(triple_result["p"], triple_result["r"], triple_result["f"]))

    return triple_result, entity_result, test_loss / steps
  

def map_origin_word_to_bert(words,tokenizer,current_idx=0):
    bep_dict = {}
    for word_idx, word in enumerate(words):
        bert_word = tokenizer.tokenize(word)
        word_len = len(bert_word)
        bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
        current_idx = current_idx + word_len
    return bep_dict

def load_demo_image(image_size,img_path):
    
    raw_image = Image.open(img_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0)
    return image

def ner_label_transform(ner_label, word_to_bert):
    new_ner_labels = []
    for i in range(0, len(ner_label), 3):
        # +1 for [CLS]
        sta = word_to_bert[ner_label[i]][0] + 1
        end = word_to_bert[ner_label[i + 1]][0] + 1
        new_ner_labels += [sta, end, ner_label[i + 2]]
    return new_ner_labels

def rc_label_transform(rc_label, word_to_bert):
    new_rc_labels = []
    for i in range(0, len(rc_label), 3):
        # +1 for [CLS]
        e1 = word_to_bert[rc_label[i]][0] + 1
        e2 = word_to_bert[rc_label[i + 1]][0] + 1
        new_rc_labels += [e1, e2, rc_label[i + 2]]
    return new_rc_labels

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="JMERE", type=str, 
                        help="which dataset to use")   # 数据集选择

    parser.add_argument("--hidden_size", default=300, type=int,
                        help="number of hidden neurons in the model")   # 隐藏神经元个数

    parser.add_argument("--bert_local_path", default="bert-base-cased", type=str, 
                        help="BERT  pretrained embedding")    # bert模型路径
    
    parser.add_argument("--blip_local_path", default="Blip/MODELS/model_base.pth", type=str, 
                        help="BLIP pretrained embedding")    # blip模型路径
    
    parser.add_argument("--Blip",default="retrieval",type=str,
                        help="about the Blip choise")
    
    parser.add_argument("--sim_mode", default="itc", type=str,
                        help="itc or itm")  

    parser.add_argument("--weight_decay", default=0., type=float,
                        help="weight decaying rate")   # 权重衰减  抑制过拟合
    
    
    parser.add_argument("--img_size", default=384, type=int,
                        help="image size of the image processing module")   # 图像大小

    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate for input word embedding")  # dropout层 张量input的值随机置0的概率

    parser.add_argument("--dropconnect", default=0.1, type=float,
                        help="dropconnect rate for partition filter layer")  # 分区过滤层的断连接率

    parser.add_argument("--clip", default=0.25, type=float,
                        help="grad norm clipping to avoid gradient explosion")

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)  # GPU选择

    ner2idx = json.load(open("datasets/" + args.data + "/ner2idx.json", "r"))

    rel2idx = json.load(open("datasets/" + args.data + "/rel2idx.json", "r"))
        
    idx2ner = {v:k for k,v in ner2idx.items()}
    idx2rel = {v:k for k,v in rel2idx.items()}
    
    test_datas = json.load(open('datasets/JMERE/JMERE_new2/new_test_triples.json','r'))
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_local_path,do_lower_case=False)

    model = TESGA(args, 768, ner2idx, rel2idx)  # 模型初始化    
    model.load_state_dict(torch.load('save2/JMERE-final1/JMERE-final1.pt'))   
    logger.info("Load model successful!") 
    model.to(device)
    model.eval()
    
    # 计算参数量大小
    params = list(model.parameters())
    k = 0
    # print(params)
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))
    print(k*4/1024/1024)
    # 计算参数量大小
     
    metric = micro(rel2idx, ner2idx)   # 

    results_list = []
    
    total_entity_num = [0,0,0]
    total_triple_num = [0,0,0]
    
    time_list = []
    for dic in test_datas:
        sent = dic['text']
        text = sent.split(' ')
        img_id = dic['img_id']
        triple_list = dic['triple_list']
        ent_pairs = dic['ent_pair_list']
        
        itc_score = dic["itc_score"]
        itc_ = itc_score
        word_to_bep = map_origin_word_to_bert(text,tokenizer)
        ner_label = []
        rc_label = []
        for index, trip in enumerate(triple_list):  
            subj = ent_pairs[index][0][0]
            subj1 = ent_pairs[index][0][1]-1
            obj = ent_pairs[index][1][0]
            obj1 = ent_pairs[index][1][1]-1
            
            rel = trip[1]
            if subj not in ner_label:
                ner_label +=[subj,subj1,trip[3]]
            if obj not in ner_label:
                ner_label +=[obj,obj1,trip[4]]
            rc_label+=[subj,obj,rel]
        
        ner_label = ner_label_transform(ner_label,word_to_bep)
        rc_label = rc_label_transform(rc_label,word_to_bep)
        bert_words = tokenizer.tokenize(sent) 
        bert_len = len(bert_words)+2
        ner_label = [gen_ner_labels(ner_label,bert_len,ner2idx)]
        ner_label = torch.stack(ner_label,dim=2).to(device)
        rc_label = [gen_rc_labels(rc_label, bert_len, rel2idx)]
        rc_label = torch.stack(rc_label,dim=2).to(device)
           
        img_p = os.path.join("datasets/" + args.data +"/JMERE_imgs/test/", img_id)
        image = load_demo_image(args.img_size,img_p).to(device)
        
        sent_bert_ids = tokenizer(text, return_tensors="pt",padding='longest',is_split_into_words=True)['input_ids'].tolist()[0]
        sent_bert_str = []
        for i in sent_bert_ids:
            sent_bert_str.append(tokenizer.convert_ids_to_tokens(i))
        
        bert_len = len(sent_bert_str)
        
        mask = torch.ones(bert_len, 1).to(device)
        itc_score= torch.Tensor([itc_score]).unsqueeze(1).to(device)
        
        import time
        t1 = time.time()
        
        ner_pred, re_pred = model(text, image, sent,itc_score, mask,token_pos=None)
        time_list.append(time.time()-t1)
        
        entity_num = metric.count_ner_num(ner_pred, ner_label)
        triple_num = metric.count_num(ner_pred, ner_label, re_pred, rc_label)

        for i in range(len(entity_num)):
            total_entity_num[i] += entity_num[i]
        for i in range(len(triple_num)):
            total_triple_num[i] += triple_num[i]
        
        ner_pred = torch.where(ner_pred>=0.5, torch.ones_like(ner_pred), torch.zeros_like(ner_pred))
        re_pred = torch.where(re_pred>=0.5, torch.ones_like(re_pred), torch.zeros_like(re_pred))
        
        entity = (ner_pred == 1).nonzero(as_tuple=False).tolist()
        relation = (re_pred == 1).nonzero(as_tuple=False).tolist()
        
        word_to_bep = map_origin_word_to_bert(text,tokenizer,current_idx=1)
        bep_to_word = {word_to_bep[i][0]:i for i in word_to_bep.keys()}
        
        entity_names = {}
        results_elist = []
        for en in entity:
            type = idx2ner[en[3]]
            start = None
            end = None
            if en[0] in bep_to_word.keys():
                start = bep_to_word[en[0]]
            if en[1] in bep_to_word.keys():
                end = bep_to_word[en[1]]
            if start == None or end == None:
                continue

            entity_str = " ".join(text[start:end+1])
            entity_names[entity_str] = start
            results_elist.append({entity_str:type})
            # print("entity_name: {}, entity type: {}".format(entity_str, type))

        results_rlist = []
        for re in relation:
            type = idx2rel[re[3]]

            e1 = None
            e2 = None

            if re[0] in bep_to_word.keys():
                e1 = bep_to_word[re[0]]
            if re[1] in bep_to_word.keys():
                e2 = bep_to_word[re[1]]
            if e1 == None or e2 == None:
                continue

            subj = None
            obj = None

            for en, start_index in entity_names.items():
                if en.startswith(text[e1]) and start_index == e1:
                    subj = en
                if en.startswith(text[e2]) and start_index == e2:
                    obj = en

            if subj == None or obj == None:
                continue

            results_rlist.append([subj, type, obj])
            # print("triple: {}, {}, {}".format(subj, type, obj))
        results_list.append({"text":sent,'img_id':img_id, 'target_list':triple_list,
                             'pred_entity':results_elist, 'pred_triple':results_rlist,"itc_score":itc_})
        
    triple_result = f1(total_triple_num)
    entity_result = f1(total_entity_num)
    logger.info("entity: p={:.4f}, r={:.4f}, f={:.4f}".format(entity_result["p"], entity_result["r"], entity_result["f"]))
    logger.info("triple: p={:.4f}, r={:.4f}, f={:.4f}".format(triple_result["p"], triple_result["r"], triple_result["f"]))
    print(sum(time_list)/len(time_list))
        
    with open('results/JMEREtest_results.json','w') as json_file:
        json.dump(results_list, json_file,indent=4)
    
    json_file.close()