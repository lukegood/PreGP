# To use multi-card training, you need to add available devices here.
export CUDA_VISIBLE_DEVICES=0  

accelerate launch \
    --main_process_port 12345 \
    --multi_gpu \
    ./finetuning.py \
        --big_batch 4 \
        --cut_length 1288 \
        --d_embedding 768 \
        --device 'cuda' \
        --end_index 300 \
        --special_word_size 5 \
        --lr 4e-5 \
        --epoch 20  \
        --kmer_k 3 \
        --load_model_name 'sample.pth' \
        --cvf_path './demo_data/cvf.csv' \
        --phe_path './demo_data/phenotyp.csv' \
        --pretrain_model_path './pretrain_model' \
        --fine_tuning_model_path './fine_tuning_model' \
        --geno_path './demo_data/genotype.csv' \
        --run_log_path "./run_log" \
        --vocab_path "./vocab" \
        --n_folds 10 \
        --gradient_accumulation_steps 4 \
        --premodel_vocab_size 32 \
        --vocab_name "sample.txt" \
        --pred_save_path "./pred_results" \
        --bag_num 1 \
        --reserved_memory 23000 \
        --random_seed 1234 \
        --unfreeze_from_layer 11