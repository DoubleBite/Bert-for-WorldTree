# Bert for WorldTree

This repository implement a BERT-based QA model for the multiple choice science questions in the [WorldTree corpus](http://cognitiveai.org/explanationbank/). 

Note that the model only handles the input consisting of **question-choice pairs** right now (without considering the supporting facts retrieved by IR modules). 


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
{"id": "Mercury_SC_LBS10276", "question": "Why do we see different stars in the sky at different times of the year?", "choices": ["The stars are revolving around the Sun.", "The Moon is revolving around Earth.", "Earth is revolving around the Sun.", "The stars are revolving around Earth."], "answer": 2, "logits": [-2.593813180923462, -4.058340072631836, 3.476123571395874, -5.399872779846191], "prediction": 2, "loss": 0.0029809109400957823, "correct": true}
{"id": "Mercury_SC_LBS10619", "question": "What remains in the same location in the sky of the Northern Hemisphere each night?", "choices": ["the Sun", "the Little Dipper", "the North Star", "the Moon"], "answer": 2, "logits": [6.39345121383667, 4.902435302734375, 6.1293768882751465, 0.30965086817741394], "prediction": 0, "loss": 0.954888641834259, "correct": false}
```


## Diagram
<img src="https://i.imgur.com/ebKj8MP.png" width="500">
