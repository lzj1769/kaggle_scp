export CUDA_VISIBLE_DEVICES=1

python train.py --valid_cell_type nk --deep_tf v4 --use_rna_pca &
python train.py --valid_cell_type t_cd4 --deep_tf v4 --use_rna_pca &
python train.py --valid_cell_type t_cd8 --deep_tf v4 --use_rna_pca &
python train.py --valid_cell_type t_reg --deep_tf v4 --use_rna_pca &
