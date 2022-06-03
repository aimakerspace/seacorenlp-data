/* Configuration for Higher-order Coreference Resolution with
   Coarse-to-fine Inference (Lee et al., 2018) with BERT-based models */

// Paths
local train_data_path = std.extVar('train_data_path');
local validation_data_path = std.extVar('validation_data_path');
local test_data_path = std.extVar('test_data_path');

// Dataset Reader
local dataset_reader = std.extVar('dataset_reader');

// Embeddings
local model_name = std.extVar('model_name');
local embedding_dim = std.parseInt(std.extVar('embedding_dim'));
local max_segment_length = std.parseInt(std.extVar('max_segment_length'));
local freeze = std.extVar('freeze');
local train_parameters = if freeze == 'true' then false else true;
local lexical_dropout = std.parseJson(std.extVar('lexical_dropout'));

// Model
local max_span_width = std.parseInt(std.extVar('max_span_width'));
local feature_size = std.parseInt(std.extVar('feature_size'));

local contextualize_embeddings = std.extVar('contextualize_embeddings');
local lstm_hidden_dim = std.parseInt(std.extVar('lstm_hidden_dim'));
local lstm_layers = std.parseInt(std.extVar('lstm_layers'));
local lstm_dropout = std.parseJson(std.extVar('lstm_dropout'));

local ffnn_layers = std.parseInt(std.extVar('ffnn_layers'));
local ffnn_hidden_dim = std.parseInt(std.extVar('ffnn_hidden_dim'));
local ffnn_dropout = std.parseJson(std.extVar('ffnn_dropout'));
local ffnn_activation = "relu";

local lstm_span_embedding_dim = 4 * lstm_hidden_dim + embedding_dim + feature_size;
local pass_through_span_embedding_dim = 3 * embedding_dim + feature_size;
local span_embedding_dim = if contextualize_embeddings == 'true' then lstm_span_embedding_dim else pass_through_span_embedding_dim;
local span_pair_embedding_dim = 3 * span_embedding_dim + feature_size;

// Training
local num_epochs = std.parseInt(std.extVar('num_epochs'));
local patience = std.parseInt(std.extVar('patience'));
local lr = std.parseJson(std.extVar('lr'));
local gpu = std.parseInt(std.extVar('gpu'));

// Modules
local transformer_token_indexer(huggingface_name) = {
    "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": huggingface_name,
        "max_length": max_segment_length
    }
};
local transformer_text_field_embedder(huggingface_name) = {
    "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": huggingface_name,
            "train_parameters": train_parameters,
            "max_length": max_segment_length
        }
    }
};
local pass_through_context_layer = {
  "type": "pass_through",
  "input_dim": embedding_dim
};
local lstm_context_layer = {
  "type": "lstm",
  "bidirectional": true,
  "input_size": embedding_dim,
  "hidden_size": lstm_hidden_dim,
  "num_layers": lstm_layers,
  "dropout": lstm_dropout
};
local context_layer = if contextualize_embeddings == 'true' then lstm_context_layer else pass_through_context_layer;

// Configuration
{
  "dataset_reader": {
    "type": dataset_reader,
    "token_indexers": transformer_token_indexer(model_name),
    "max_span_width": max_span_width
  },
  "train_data_path": train_data_path,
  "validation_data_path": validation_data_path,
  "test_data_path": test_data_path,
  "evaluate_on_test": true,
  "model": {
    "type": "coref",
    "text_field_embedder": transformer_text_field_embedder(model_name),
    "context_layer": context_layer,
    "mention_feedforward": {
        "input_dim": span_embedding_dim,
        "num_layers": ffnn_layers,
        "hidden_dims": ffnn_hidden_dim,
        "activations": ffnn_activation,
        "dropout": ffnn_dropout
    },
    "antecedent_feedforward": {
        "input_dim": span_pair_embedding_dim,
        "num_layers": ffnn_layers,
        "hidden_dims": ffnn_hidden_dim,
        "activations": ffnn_activation,
        "dropout": ffnn_dropout
    },
    "initializer": {
        "regexes": [
            [".*linear_layers.*weight", {"type": "xavier_normal"}],
            [".*scorer._module.weight", {"type": "xavier_normal"}],
            ["_distance_embedding.weight", {"type": "xavier_normal"}],
            ["_span_width_embedding.weight", {"type": "xavier_normal"}],
            ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
            ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
        ]
    },
    "lexical_dropout": lexical_dropout,
    "feature_size": feature_size,
    "max_span_width": max_span_width,
    "spans_per_word": 0.4,
    "max_antecedents": 50,
    "coarse_to_fine": true,
    "inference_order": 2
  },

  "data_loader": {
      "batch_sampler": {
          "type": "bucket",
          "sorting_keys": ["text"],
          "batch_size": 1,
          "padding_noise": 0.0,
      }
  },
  "trainer": {
    "cuda_device": gpu,
    "num_epochs": num_epochs,
    "patience" : patience,
    "validation_metric": "+coref_f1",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": lr,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    },
  }
}
