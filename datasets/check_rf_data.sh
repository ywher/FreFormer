split='val'  # 'train' 'val'

rf_path="../data/xinhua_kidney_rf/$split/rf"
python check_rf_data.py \
--rf_dir $rf_path \
--output_file ${split}_failed_rf_files.txt \