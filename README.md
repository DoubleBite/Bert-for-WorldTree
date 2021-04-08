# Bert for WorldTree

This repository implement a BERT-based QA model for the multiple choice science questions in the WorldTree corpus. 

Note that the model only handles the input consisting of **question and choice pairs** right now (without considering the supporting facts retrieved from IR modules). 


|                 | Accuracy      |
| -------------   |:-------------:|
|BERT-base-uncased| 47.25%        |
|  RoBERTa-large  | 58.19%        |




## Run the Experiments

The experiments are:
1. Trained on `data/questions/questions.train.tsv`
2. Validated on `data/questions/questions.dev.tsv`
3. Evaluated on `data/questions/questions.test.tsv`
4. `result_dir` is the directory to store the experimental results.


### 1. Run the BERT model
```
bash run_experiment.sh result_dir bert 
```

### 2. Run the RoBERTa-large model
```
bash run_experiment.sh result_dir roberta
```

### 3. Calculate the accuracy for the multiple choice questions

```
python evaluate.py result_dir
```



### 4. Example Predictions

The prediction results are stored in `result_dir/predictions.jsonl`, and they look like this

```

```



## Diagram
<img src="https://i.imgur.com/ebKj8MP.png" width="500">
