# Parameter Description
## Pretrain Phase
| Parameter | Description |
|-----------|-------------|
| `batch` | Batch size for training |
| `cut_length` | Sequence cut length |
| `d_embedding` | Dimension of token embeddings |
| `end_index` | Maximum number of SNPs to use from the genotype file |
| `mask_ratio` | Proportion of tokens to mask during MLM training |
| `lr` | Learning rate for optimizer |
| `epoch` | Number of training epochs |
| `geno_path` | Path to genotype data CSV file |
| `pretrain_model_path` | Directory to save pre-trained model checkpoints |
| `run_log_path` | Directory for training logs |
| `vocab_path` | Directory containing vocabulary files |
| `checkpoint_save_path` | Directory to save training checkpoints |
| `checkpoint_load_file_path` | Directory to load checkpoints from |
| `save_interval` | Save checkpoint every N steps |
| `eval_freq` | Evaluate model every N batches |
| `reserved_memory` | Reserved GPU memory in MB |
| `kmer_k` | K-mer size for tokenization  |

## Finetuning Phase
| Parameter | Description |
|-----------|-------------|
| `big_batch` | batch size  |
| `cut_length` | Sequence cut length  |
| `d_embedding` | Dimension of token embeddings |
| `end_index` | Maximum number of SNPs to use from the genotype file |
| `special_word_size` | Size of special vocabulary |
| `lr` | Learning rate for optimizer |
| `epoch` | Number of training epochs |
| `kmer_k` | K-mer size for tokenization |
| `load_model_name` | Filename of the pre-trained model checkpoint to load |
| `cvf_path` | Path to cross-validation folds CSV file |
| `phe_path` | Path to phenotype data CSV file |
| `pretrain_model_path` | Directory containing the pre-trained model |
| `fine_tuning_model_path` | Directory to save fine-tuned model checkpoints |
| `geno_path` | Path to genotype data CSV file |
| `run_log_path` | Directory for training logs |
| `vocab_path` | Directory containing vocabulary files |
| `n_folds` | Number of cross-validation folds |
| `gradient_accumulation_steps` | Number of steps to accumulate gradients before updating weights |
| `premodel_vocab_size` | Vocabulary size of the pre-trained model |
| `vocab_name` | Name of the vocabulary file |
| `pred_save_path` | Directory to save prediction results |
| `reserved_memory` | Reserved GPU memory in MB |
| `random_seed` | Random seed for reproducibility |
| `unfreeze_from_layer` | Layer index from which to unfreeze parameters |