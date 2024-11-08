python3 main.py \
    --data JMERE \
    --batch_size  16 \
    --eval_batch_size 16 \
    --lr 0.00003 \
    --output_file JMERE-final1 \
    --eval_metric micro \
    --epoch 60 \
    --bert_local_path bert-base-cased \
    --blip_local_path  Blip/MODELS/model_base.pth \
    --sim_mode  itc \
    --do_eval \
    --do_train 
    
    
    