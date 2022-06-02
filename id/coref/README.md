# COIN: A Large-scale Benchmark Dataset for Coreference Resolution in the Indonesian Language

This repository contains the COIN dataset for coreference resolution in Indonesian.

## TSV Format

The TSV files (train/dev/test) contain the coreference resolution data in the CoNLL-U columnar format.
An example is as follows:

| Paragraph ID - Token Index | Character Span Indices | Token     | Labels                |
|----------------------------|------------------------|-----------|-----------------------|
| 1-1                        | 0-4                    | Adam      | PROPER[1]             |
| 1-2                        | 5-11                   | speaks    | _                     |
| 1-3                        | 12-22                  | Indonesian| _                     |
| 1-4                        | 22-23                  | .         | _                     |
| 1-5                        | 24-26                  | He        | IDENT[1_2]\|PRONOUN[2]|
| 1-6                        | 27-31                  | does      | _                     |
| 1-7                        | 32-35                  | not       | _                     |
| 1-8                        | 36-41                  | speak     | _                     |
| 1-9                        | 42-48                  | French    | _                     |
| 1-10                       | 49-50                  | .         | _                     |


The labels column (Column 4) can contain both mention labels and coreference link labels.
If there is more than one label, they are separated by a pipe |.
Mention labels consist of the mention type followed by square brackets containing the mention ID.
Coreference link labels consist of the coreference type followed by square brackets containing the IDs of the two coreferent mentions.
In the example above, the token `Adam` is labeled as `PROPER[1]`, meaning that it is a proper noun mention,
and `He` is a pronoun, `PRONOUN[2]`, that is linked to `PROPER[1]` via the `IDENT` relation.

## Data Preprocessing

In order to preprocess the data in the same manner as is described in the paper, ensure your environment has `jsonlines` and `tqdm` installed, then run the following command:

```bash
bash preprocess_data.sh
```

You should get `train.jsonl`, `dev.jsonl` and `test.jsonl`.

The file format is structured as such:

```json
{
  "text": "Adam speaks Indonesian. He does not speak French.",
  "tokens": ["Adam", "speaks", "Indonesian", ".", "He", "does", "not", "speak", "French", "."],
  "corefs": [
    {
      "start": 0,
      "end": 1,
      "text": "Adam",
      "type": "PROPER",
      "label": 0
    },
    {
      "start": 4,
      "end": 5,
      "text": "He",
      "type": "PRONOUN",
      "label": 0
    }
  ]
}
```

The `label` property denotes an arbitrary integer that is used to identify which mentions corefer (i.e. if they share the same label, they are coreferent.)

As explained in the paper, mentions linked by an APPOS relation are concatenated, while mentions linked via EXAPPOS relations are dropped.

## Model Training/Evaluation

We trained and evaluated the c2f-coref models in our paper using the AllenNLP framework. For the best configuration, please run the training with the following command (requires GPU):

```bash
train_data_path=train.jsonl validation_data_path=dev.jsonl test_data_path=test.jsonl dataset_reader=coref-id \     model_name=$BERT_MODEL_NAME embedding_dim=1024 max_segment_length=512 freeze=false lexical_dropout=0.2 \
max_span_width=30 feature_size=20 contextualize_embeddings=false \
ffnn_layers=2 ffnn_hidden_dim=1024 ffnn_dropout=0.2 num_epochs=40 patience=10 lr=3e-4 gpu=$GPU_ID \
allennlp train coref.jsonnet -s $TRAINED_MODEL_PATH --include-package training
```

`$BERT_MODEL_NAME` should be a reference to a model name on HuggingFace, such as `xlm-roberta-base`.

## License

The COIN dataset is made available under the CC BY-SA 4.0 License.

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
