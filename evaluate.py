import sys
import json


def evaluate_predictions(prediction_file_path):
    
    # Read the predictions
    results = []
    with open(prediction_file_path, 'r') as f:
        for line in f.readlines():
            results.append(json.loads(line))
    
    # Calculate accuracy
    num_correct_answer = 0
    for result in results:
        if result["correct"] is True:
            num_correct_answer +=1
    
    accuracy = num_correct_answer/len(results)
    
    return accuracy


if __name__=="__main__":
    
    result_dir = sys.argv[1]
    prediction_file_path = f"{result_dir}/predictions.jsonl"
    
    accuracy = evaluate_predictions(prediction_file_path)
    print(accuracy)
    