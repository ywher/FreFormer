# xinhua kidney dataset
train_img_dir="./data/xinhua_kidney/train/image"
train_mask_dir="./data/xinhua_kidney/train/mask"
val_img_dir="./data/xinhua_kidney/val/image"
val_mask_dir="./data/xinhua_kidney/val/mask"

CUDA_VISIBLE_DEVICES=1 python train_mix_kidney.py \
--train_img_dir $train_img_dir \
--train_mask_dir $train_mask_dir \
--val_img_dir $val_img_dir \
--val_mask_dir $val_mask_dir


# xinhua kidney rf dataset
# train_img_dir="./data/xinhua_kidney_rf/train/image"
# train_mask_dir="./data/xinhua_kidney_rf/train/mask"
# train_rf_dir="./data/xinhua_kidney_rf/train/rf"
# val_img_dir="./data/xinhua_kidney_rf/val/image"
# val_mask_dir="./data/xinhua_kidney_rf/val/mask"
# val_rf_dir="./data/xinhua_kidney_rf/val/rf"

# CUDA_VISIBLE_DEVICES=0 python train_mix_kidney.py \
# --train_img_dir $train_img_dir \
# --train_mask_dir $train_mask_dir \
# --train_rf_dir $train_rf_dir \
# --val_img_dir $val_img_dir \
# --val_mask_dir $val_mask_dir \
# --val_rf_dir $val_rf_dir \
# --use_rf