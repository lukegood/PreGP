# To use multi-card training, you need to add available devices here.
export CUDA_VISIBLE_DEVICES=0  

accelerate launch \
    --main_process_port 12345 \
    --multi_gpu \
    --mixed_precision=fp16 \
    ./pretrain.py \
        --batch 2 \
        --cut_length 1288 \
        --d_embedding 768 \
        --device 'cuda' \
        --end_index 300 \
        --special_word_size 5 \
        --mask_ratio 0.15 \
        --lr 4e-5 \
        --epoch 100  \
        --geno_path './demo_data/cvf.csv' \
        --pretrain_model_path "./pretrain_model" \
        --run_log_path "./run_log" \
        --vocab_path "./vocab" \
        --checkpoint_save_path "./check_point_save/sample" \
        --checkpoint_load_file_path "./check_point_save/sample" \
        --save_interval 20000 \
        --eval_freq 2000 \
        --reserved_memory 12000 \
        --bag_num 1 \
        --kmer_k 3