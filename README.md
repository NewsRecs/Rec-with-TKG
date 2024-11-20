# Rec-with-TKG (incompleted)
├── README.md
└── Adressa_preprocessing
     ├── App.jsx
     ├── index.jsx
     ├── api
     │     └── mandarinAPI.js
     ├── data
     │     ├── history
     │     │     ├── 20170101
     │     │     ├── ...
     │     │     ├── 20170205
     │     │     ├── news.tsv
     │     │     └── news(raw).tsv
     │     ├── train
     │     │     ├── 20170205
     │     │     ├── ...
     │     │     ├── 20170212
     │     │     ├── behaviors.tsv
     │     │     ├── behaviors(raw).tsv
     │     │     ├── news.tsv
     │     │     └── news(raw).tsv
     │     └── user.tsv
     ├── gpt_finetuning_data
     │     ├── train_negative.jsonl
     │     ├── train_positive.jsonl
     │     ├── val_negative.jsonl
     │     └── val_positive.jsonl
     ├── result
     │     ├── error_detect
     │     │     ├── negative_detected.txt
     │     │     ├── negative_finetuned_detected.txt
     │     │     ├── positive_detected.txt
     │     │     └── positive_finetuned_detected.txt
     │     ├── experiment_result
     │     │     ├── negative_metrics.txt
     │     │     ├── negative_finetuned_metrics.txt
     │     │     ├── positive_metrics.txt
     │     │     └── positive_finetuned_metrics.txt
     │     └── gpt_result
     │           ├── negative.txt
     │           ├── negative_finetuned.txt
     │           ├── positive.txt
     │           └── positive_finetuned.txt
     ├── .env
     ├── 0. debugging.ipynb
     ├── 0. prompt_tokens_check.ipynb
     ├── 1. creat_prompts.ipynb
     ├── 2. generate_json.ipynb
     ├── 3. experiment.ipynb
     └── 4. create_metrics
