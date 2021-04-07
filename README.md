# Bert for WorldTree

This repository implement a BERT-based QA model for the multiple choice science questions in the WorldTree corpus. 

Note that the model only handles the input consisting of **question and choice pairs** right now (without considering the supporting facts retrieved from IR modules). 


|                 | Accuracy      |
| -------------   |:-------------:|
|BERT-base-uncased| right-aligned |
|  RoBERTa-large  | centered      |



## Diagram
<img src="https://i.imgur.com/ebKj8MP.png" width="500">


## Run the Experiments

+ Run the BERT model
```
bash run_experiment.sh some_result_dir bert 
```

+ Run the RoBERTa-large model
```
bash run_experiment.sh some_result_dir roberta
```


