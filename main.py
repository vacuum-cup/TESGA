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
from model.Multichan_BLIP_retrieval_co import TESGA
from dataloader.dataloader import dataloader
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S',  # 日期格式
                    level=logging.INFO)   # 级别
logger = logging.getLogger(__name__)   # 日志处理器


class WarmUpCosineAnnealingLR(LambdaLR): 
    # 训练过程中 先warmup 后CosineAnnealingLR 变化学习率
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0.001, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_epoch):
        if current_epoch < self.warmup_epochs:
            return float(current_epoch) / float(max(1, self.warmup_epochs))
        progress = (current_epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
        return self.eta_min + 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))) * (1.0 - self.eta_min)


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
  
def seed_everything(seed=2021):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="JMERE", type=str, required=True,
                        help="which dataset to use")   # 数据集选择

    parser.add_argument("--epoch", default=50, type=int,
                        help="number of training epoch")    # 训练迭代次数

    parser.add_argument("--hidden_size", default=300, type=int,
                        help="number of hidden neurons in the model")   # 隐藏神经元个数

    parser.add_argument("--batch_size", default=16, type=int,
                        help="number of samples in one training batch")   #模型训练时的batch size
    
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="number of samples in one testing batch")    #测试过程的batch size
    
    parser.add_argument("--do_train", action="store_true",
                        help="whether or not to train from scratch")   #是否进行训练的标志

    parser.add_argument("--do_eval", action="store_true",
                        help="whether or not to evaluate the model")    # 是否进行evaluate 的标志

    parser.add_argument("--bert_local_path", default="/home/users/wgx/RE_NER/BERT_DIR/bert-base-cased", type=str, 
                        help="BERT  pretrained embedding")    # bert模型路径
    
    parser.add_argument("--blip_local_path", default="/home/users/wgx/Mult_m_NRE/multi-joint-ner-re/Blip/MODELS/model_base.pth", type=str, 
                        help="BLIP pretrained embedding")    # blip模型路径
    
    parser.add_argument("--Blip",default="retrieval",type=str,
                        help="about the Blip choise")

    parser.add_argument("--eval_metric", default="micro", type=str,
                        help="micro f1 or macro f1")   # 两种不同的F1得分计算方法  
    
    parser.add_argument("--sim_mode", default="itc", type=str,
                        help="itc or itm")  

    parser.add_argument("--lr", default=0.00003, type=float,
                        help="initial learning rate")    # 学习率

    parser.add_argument("--weight_decay", default=0., type=float,
                        help="weight decaying rate")  
    
    parser.add_argument("--linear_warmup_rate", default=0.0, type=float,
                        help="warmup at the start of training") 
    
    parser.add_argument("--seed", default=2023, type=int,
                        help="random seed initiation")   
    
    parser.add_argument("--img_size", default=384, type=int,
                        help="image size of the image processing module")   

    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate for input word embedding")  

    parser.add_argument("--dropconnect", default=0.1, type=float,
                        help="dropconnect rate for partition filter layer") 

    parser.add_argument("--steps", default=100, type=int,
                        help="show result for every 100 steps")

    parser.add_argument("--output_file", default="jmere", type=str, required=True,
                        help="name of result file")

    parser.add_argument("--clip", default=0.25, type=float,
                        help="grad norm clipping to avoid gradient explosion")

    args = parser.parse_args()
    
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1)  # GPU选择
    

    output_dir = "save/" + args.output_file
    # os.mkdir(output_dir)

    # 将超参数和一些训练过程参数 写入日志中
    logger.addHandler(logging.FileHandler(output_dir + "/" + args.output_file + ".log", 'w'))
    logger.info(sys.argv)
    logger.info(args)

    saved_file = save_results(output_dir + "/" + args.output_file + ".txt", header="# epoch \t train_loss \t  dev_loss \t test_loss \t dev_ner \t dev_rel \t test_ner \t test_rel")
    model_file = args.output_file + ".pt"
    result_flle = "final.pt"
       
    with open("datasets/" + args.data + "/ner2idx.json", "r") as f:   # 实体 to id
        ner2idx = json.load(f)
    with open("datasets/" + args.data + "/rel2idx.json", "r") as f:    # 关系 to id
        rel2idx = json.load(f)
 
    # 数据加载
    train_batch, test_batch, dev_batch = dataloader(args, ner2idx, rel2idx) 

    if args.do_train:
        logger.info("------Training------")
        input_size = 768 

        model = TESGA(args, input_size, ner2idx, rel2idx)     
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 优化器
        
        warmup_epochs = 5
        # 指定学习率最小值
        eta_min = 0.00001    
        scheduler = WarmUpCosineAnnealingLR(optimizer, warmup_epochs, args.epoch, eta_min=eta_min)
        
        if args.eval_metric == "micro":
            metric = micro(rel2idx, ner2idx)   # 
        else:
            metric = macro(rel2idx, ner2idx)

        BCEloss = loss()
        best_result = 0
        triple_best = None
        entity_best = None

        for epoch in range(args.epoch):
            steps, train_loss = 0, 0

            total_triple_num = [0, 0, 0]
            total_entity_num = [0, 0, 0]

            if args.eval_metric == "macro":
                total_triple_num = total_triple_num * len(rel2idx)
                total_entity_num = total_entity_num * len(ner2idx)

            model.train()  # 训练标志
            scheduler.step()  ## 调整学习率
            
            for data in tqdm(train_batch):  # tqdm 显示进度条
                
                steps+=1
                optimizer.zero_grad()

                text = data[0]  
                sim_scores = torch.Tensor(data[1]).unsqueeze(1).to(device)
                ner_label = data[2].to(device)   # 实体标签
                re_label = data[3].to(device)   # 关系标签
                mask = data[4].to(device)  # 对每个句子，我们都认为其句长为 max_len ,因此 对于< max_len的句子 需要掩盖掉多余信息
                images = data[5].to(device)  
                sents = data[6] 
                # token_pos = data[7].to(device)
                token_pos = None
                
                ner_pred, re_pred = model(text, images, sents,sim_scores, mask,token_pos)
                loss = BCEloss(ner_pred, ner_label, re_pred, re_label)

                loss.backward()

                train_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.clip)
                optimizer.step()

                entity_num = metric.count_ner_num(ner_pred, ner_label)
                triple_num = metric.count_num(ner_pred, ner_label, re_pred, re_label)

                for i in range(len(entity_num)):
                    total_entity_num[i] += entity_num[i]
                for i in range(len(triple_num)):
                    total_triple_num[i] += triple_num[i]


                if steps % args.steps == 0:
                    logger.info("Epoch: {}, step: {} / {}, loss = {:.4f}, lr={:.8f}".format
                                (epoch, steps, len(train_batch), train_loss / steps,optimizer.param_groups[0]['lr']))


            triple_result = f1(total_triple_num)
            entity_result = f1(total_entity_num)
            logger.info("------ Training Set Results ------")
            logger.info("loss : {:.4f}".format(train_loss / steps))
            logger.info("entity: p={:.4f}, r={:.4f}, f={:.4f}".format(entity_result["p"], entity_result["r"], entity_result["f"]))
            logger.info("triple: p={:.4f}, r={:.4f}, f={:.4f}".format(triple_result["p"], triple_result["r"], triple_result["f"]))

            if args.do_eval:
                model.eval()
                logger.info("------ Testing ------")
                dev_triple, dev_entity, dev_loss = evaluate(dev_batch, rel2idx, ner2idx, args, "dev")
                test_triple, test_entity, test_loss = evaluate(test_batch, rel2idx, ner2idx, args, "test")
                average_f1 = dev_triple["f"] + dev_entity["f"]

                if epoch == 0 or average_f1 > best_result:
                    best_result = average_f1
                    triple_best = test_triple
                    entity_best = test_entity
                    torch.save(model.state_dict(), output_dir + "/" + model_file)
                    logger.info("Best result on dev saved!!!")
                
                if epoch == args.epoch-1:  # 保存最后一个epoch的模型
                    torch.save(model.state_dict(), output_dir + "/" + result_flle)
                    logger.info("final  model saved!!!")


                saved_file.save("{} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}".format(epoch, train_loss/steps, dev_loss, test_loss, dev_entity["f"],
                                                   dev_triple["f"], test_entity["f"], test_triple["f"]))

        saved_file.save("best test result ner-p: {:.4f} \t ner-r: {:.4f} \t ner-f: {:.4f} \t re-p: {:.4f} \t re-r: {:.4f} \t re-f: {:.4f} ".format(entity_best["p"],
                        entity_best["r"], entity_best["f"], triple_best["p"], triple_best["r"], triple_best["f"]))

