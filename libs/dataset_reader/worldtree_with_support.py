import logging
from typing import List, Optional, Iterable
import re
import pandas as pd
import json

from overrides import overrides
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.fields import TextField, ListField, IndexField, MetadataField
from allennlp.common.file_utils import cached_path

from libs.dataset_reader.utils import parse_raw_question, answser_to_index

logger = logging.getLogger(__name__)


@DatasetReader.register("worldtree-support")
class WorldTreeSupportReader(DatasetReader):
    """

    """

    def __init__(
        self,
        transformer_model_name: str = "roberta-large",
        topk: int = 5,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name, add_special_tokens=False
        )

        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(transformer_model_name)}

        # Get the topk supporting facts
        self.topk = topk

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """


        """
        with open(cached_path(file_path), "r") as data_file:

            logger.info(
                "Reading questions from file at: %s", file_path)

            questions = json.load(data_file)

            for question in questions:
                qid = question["id"]
                question_text = question["question"]
                supporting_facts = question["supports"]
                choices = question["choices"]
                answer = question["answer"]
                yield self.text_to_instance(qid, question_text, supporting_facts, choices, answer)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        qid: str,
        question: str,
        supporting_facts: List[str],
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

        # Combine supporting facts with questions
        # Here we're just concatenate the supporting facts to the end of the question
        supporting_facts = supporting_facts[:self.topk]
        question = " ".join([question] + supporting_facts)

        # Tokenize the question
        question_tokens = self._tokenizer.tokenize(question)

        # Tokenize the choices and concate them and the question into question-choice pairs
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
