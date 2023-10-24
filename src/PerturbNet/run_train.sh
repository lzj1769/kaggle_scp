export CUDA_VISIBLE_DEVICES=1

python train.py --valid_cell_type nk --use_deep_tf_v1 &
python train.py --valid_cell_type t_cd4 --use_deep_tf_v1 &
python train.py --valid_cell_type t_cd8 --use_deep_tf_v1 &
python train.py --valid_cell_type t_reg --use_deep_tf_v1 &
