import logging
from typing import List, Optional, Iterable

import re
import pandas as pd

from overrides import overrides
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.fields import TextField, ListField, IndexField, MetadataField
from allennlp.common.file_utils import cached_path

logger = logging.getLogger(__name__)


@DatasetReader.register("worldtree")
class WorldTreeReader(DatasetReader):
    """

    """

    def __init__(
        self,
        transformer_model_name: str = "roberta-large",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name, add_special_tokens=False
        )

        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(transformer_model_name)}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """


        """
        with open(cached_path(file_path), "r") as data_file:

            logger.info(
                "Reading questions from file at: %s", file_path)

            df = pd.read_csv(file_path, delimiter="\t")

            for _, row in df.iterrows():
                qid = row["QuestionID"]
                raw_question = row["question"]
                question, choices = _parse_question_and_choices(
                    raw_question)
                answer = row["AnswerKey"]
                answer_idx = answser_to_index(answer)
                yield self.text_to_instance(qid, question, choices, answer_idx)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        qid: str,
        question: str,
        choices: List[str],
        answer_idx: Optional[int] = None,
    ) -> Instance:

        # **A hack**
        # We need to make each question have exactly four choices to process them in batches.
        # Either drop the choice or use a non-answer one to fill.
        if len(choices) == 5:
            if answer_idx != 4:  # Just drop the last choice
                choices = choices[:-1]
            elif answer_idx == 4:  # Answer is the last, so drop the first
                choices = choices[1:]
                answer_idx -= 1
        elif len(choices) == 3:
            if answer_idx != 2:  # Use the last to fill
                choices.append(choices[-1])
            else:  # Use the first to fill
                choices.append(choices[0])

        # Base checks
        assert len(choices) == 4
        if answer_idx < 0 or answer_idx >= len(choices):
            # print(answer_idx)
            raise ValueError("Choice %d does not exist", answer_idx)

        # Tokenize the question
        question_tokens = self._tokenizer.tokenize(question)

        # Tokenize the choices and concate them with the question into question-choice pairs
        qc_pairs = []
        for choice in choices:
            choice_tokens = self._tokenizer.tokenize(choice)
            qc_pair = self._tokenizer.add_special_tokens(
                question_tokens, choice_tokens)
            qc_pairs.append(qc_pair)

        # Wrap them into AllenNLP fields
        qc_pairs = [TextField(pair, self._token_indexers)
                    for pair in qc_pairs]
        qc_pairs = ListField(qc_pairs)
        answer_idx = IndexField(answer_idx, qc_pairs)
        metadata = MetadataField({
            "id": qid,
            "question": question,
            "choices": choices
        })

        return Instance({
            "qc_pairs": qc_pairs,
            "answer_idx": answer_idx,
            "metadata": metadata,
        })


def _parse_question_and_choices(raw_question: str):
    """

        Example Usage
        ---------------------


    """
    trunks = re.split(r"\([ABCDE12345]\)", raw_question)
    trunks = [x.strip() for x in trunks]
    question, *choices = trunks

    return question, choices


def answser_to_index(answer: str):
    """
    The answer is the nth choices
    """
    mapping = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4,
    }

    return mapping[answer]
