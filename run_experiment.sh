PACKAGE="libs"
RESULT_DIR="results/$1"
TEST_PATH="data/questions/questions.test.tsv"

# Choose the experiment config 
if [ "$2" = "roberta" ]
then
    CONFIG="configs/roberta_large.jsonnet"
elif [ "$2" = "bert" ]
then
    CONFIG="configs/bert_base.jsonnet"
fi

python -m allennlp train \
    $CONFIG \
    --serialization-dir $RESULT_DIR \
    --include-package $PACKAGE \
    -f

python -m allennlp predict \
    ${RESULT_DIR}/model.tar.gz \
    $TEST_PATH \
    --include-package $PACKAGE \
    --output-file ${RESULT_DIR}/predictions.jsonl \
    --use-dataset-reader \
    --cuda-device 0 \
    --predictor worldtree
