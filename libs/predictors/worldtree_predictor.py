from typing import List, Dict, Any
from overrides import overrides
import json

from allennlp.predictors import Predictor
from allennlp.data import Instance
from allennlp.common.util import JsonDict, sanitize


@Predictor.register("worldtree")
class WorldTreePredictor(Predictor):
    """
    """

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:

        outputs = self._model.forward_on_instance(instance)

        metadata = instance["metadata"]
        label = instance["answer_idx"].human_readable_repr()
        prediction = outputs["prediction"]

        # Reorganize the outputs
        outputs = {
            "id": metadata["id"],
            "question": metadata["question"],
            "choices": metadata["choices"],
            "answer": label,
            **outputs,
            "correct": (label == prediction)
        }

        return sanitize(outputs)
