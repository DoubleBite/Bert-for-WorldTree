"""

Modified from: https://github.com/allenai/allennlp-models/blob/main/allennlp_models/mc/models/transformer_mc.py 
"""

import logging
from typing import Dict, List, Optional
from overrides import overrides

import torch

from allennlp.data import Vocabulary, TextFieldTensors

from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.seq2vec_encoders import BertPooler

from allennlp.training.metrics import CategoricalAccuracy


logger = logging.getLogger(__name__)


@Model.register("worldtree")
class TransformerWorldTree(Model):
    """

    Parameters
    ----------
    vocab : ``Vocabulary``
    transformer_model : ``str``, optional (default=``"roberta-large"``)
        This model chooses the embedder according to this setting. You probably want to make sure this matches the
        setting in the reader.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        transformer_model: str = "roberta-large",
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._text_field_embedder = PretrainedTransformerEmbedder(
            transformer_model,
            override_weights_file=override_weights_file,
            override_weights_strip_prefix=override_weights_strip_prefix,
        )
        self._text_field_embedder = BasicTextFieldEmbedder(
            {"tokens": self._text_field_embedder})
        self._pooler = BertPooler(
            transformer_model,
            override_weights_file=override_weights_file,
            override_weights_strip_prefix=override_weights_strip_prefix,
            dropout=0.1,
        )

        self._linear_layer = torch.nn.Linear(
            self._text_field_embedder.get_output_dim(), 1)
        self._linear_layer.weight.data.normal_(mean=0.0, std=0.02)
        self._linear_layer.bias.data.zero_()

        self._loss = torch.nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

    def forward(
        self,  # type: ignore
        qc_pairs: TextFieldTensors,
        answer_idx: Optional[torch.IntTensor] = None,
        metadata: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        qc_pairs : ``Dict[str, torch.LongTensor]``
            From a ``ListField[TextField]``. Contains a list of question-choice pairs to evaluate for every instance.
        answer_idx : ``Optional[torch.IntTensor]``
            From an ``IndexField``. Contains the index of the correct answer for every instance.
        metadata : `Optional[Dict[str, Any]]`
            The meta information for the questions, like question_id, original_text, and so on.
            
        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. This is only returned when `answer_idx` is not `None`.
        logits : ``torch.FloatTensor``
            The logits for every possible answer choice.
        prediction : ``List[int]``
            The index of the highest scoring choice for every instance in the batch.
        """

        # Bert embedding
        # Shape: (batch_size, num_choices, seq_length, embedding_dim)
        embedded_pairs = self._text_field_embedder(
            qc_pairs, num_wrapping_dims=1)
        batch_size, num_choices, seq_length, embedding_dim = embedded_pairs.size()

        # Flatten the choices
        # Shpae: (batch_size*num_choices, seq_length, embedding_dim)
        flattened = embedded_pairs.view(
            batch_size * num_choices,
            seq_length,
            embedding_dim,
        )

        # Get the embedding of the [CLS] for classification
        # Shpae: (batch_size*num_choices, embedding_dim)
        pooled = self._pooler(flattened)

        # Pass through a linear layer to predict the logits
        # Shpae: (batch_size*num_choices, 1)
        logits = self._linear_layer(pooled)

        # Restore the shapes
        # Shape: (batch_size, num_choices)
        logits = logits.view(
            batch_size, num_choices
        )
        prediction = logits.argmax(-1)

        # If answer_idx is passed, calculate the loss
        if answer_idx is not None:
            answer_idx = answer_idx.squeeze(1)
            loss = self._loss(
                logits, answer_idx)
            self._accuracy(logits, answer_idx)
        else:
            loss = None

        return {
            "logits": logits,
            "prediction": prediction,
            "loss": loss,
        }

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "acc": self._accuracy.get_metric(reset),
        }
