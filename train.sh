set -e

# python -m torch.distributed.launch --nproc_per_node=1 --use_env Pretrain.py --config ./configs/Pretrain.yaml --mode train
# python -m torch.distributed.launch --nproc_per_node=1 --use_env Pretrain.py --config ./configs/Pretrain_simple.yaml --mode train

python -m torch.distributed.launch --nproc_per_node=1 --use_env Pretrain.py --config ./configs/Pretrain.yaml --mode test
python -m torch.distributed.launch --nproc_per_node=1 --use_env Pretrain.py --config ./configs/Pretrain_simple.yaml --mode test

python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true

python Sharpen_unet.py --config ./configs/Sharpen_unet.yaml --mode train_valid
python Sharpen_unet.py --config ./configs/Sharpen_unet.yaml --mode test --checkpoint output/handa_book_sharpen_unet_base_inv_96/checkpoint_09.pth
python Sharpen_unet.py --config ./configs/Sharpen_unet.yaml --mode test --checkpoint output/handa_book_sharpen_unet_scale_inv_96/checkpoint_09.pth

python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --checkpoint output/finetune_single_mlm_cb/checkpoint_10.pth --text_encoder '' --mode test
python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --checkpoint output/finetune_single_mlm_np_mk50/checkpoint_48.pth --text_encoder '' --mode test
python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --checkpoint output/finetune_single_mlm_np_mk75/checkpoint_59.pth --text_encoder '' --mode test
