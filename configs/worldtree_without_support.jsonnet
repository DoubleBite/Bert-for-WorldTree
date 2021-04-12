local epochs = 20;
local batch_size = 64;

local gpu_batch_size = 4;
local gradient_accumulation_steps = batch_size / gpu_batch_size;

{
    "dataset_reader": {
        "type": "worldtree",
        "transformer_model_name": "roberta-large",
      //"max_instances": 200  // debug setting
    },
    "train_data_path": "data/questions/questions.train.tsv",
    "validation_data_path": "data/questions/questions.dev.tsv",
    "model": {
        "type": "worldtree",
        "transformer_model": "roberta-large",
    },
    "data_loader": {
        "shuffle": true,
        "batch_size": gpu_batch_size
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "weight_decay": 0.01,
            "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
            "lr": 1e-5,
            "eps": 1e-8,
            "correct_bias": true
        },
        "learning_rate_scheduler": {
            "type": "linear_with_warmup",
            "warmup_steps": 100
        },
        "num_epochs": epochs,
        "num_gradient_accumulation_steps": gradient_accumulation_steps,
        "patience": 3,
        "validation_metric": "+acc",
    },
    "random_seed": null,
    "numpy_seed": null,
    "pytorch_seed": null,
}