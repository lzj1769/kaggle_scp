export CUDA_VISIBLE_DEVICES=1

python train.py --valid_cell_type nk --epoch 100 &
python train.py --valid_cell_type t_cd4 --epoch 100 &
python train.py --valid_cell_type t_cd8 --epoch 100 &
python train.py --valid_cell_type t_reg --epoch 100 &
