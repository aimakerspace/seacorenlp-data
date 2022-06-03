python -m preprocess --input_path=train.tsv --output_path=train.jsonl --use_appos --use_aliases --remove_singletons
python -m preprocess --input_path=dev.tsv --output_path=dev.jsonl --use_appos --use_aliases --remove_singletons
python -m preprocess --input_path=test.tsv --output_path=test.jsonl --use_appos --use_aliases --remove_singletons
