dataset=amazon_us_reviews
dataset_config=Baby_v1_00

printf "Start preprocess...\n"
python preprocess_dataset.py --dataset_name ${dataset} --dataset_config_name $dataset_config
printf "End preprocess\n"
printf "Saved to %s\n" $dataset


printf "Start train..."
python run_mlm_torch.py --output_dir textsettr_out --model_name_or_path t5-small --tokenizer_name t5-small --dataset_name ${dataset} --load_from disk --do_train --do_eval --num_train_epochs 10 --overwrite_output_dir
