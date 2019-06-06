from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np

from functools import partial
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import Auc
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import Predictor
from allennlp.common import JsonDict

out_dir = "/Users/justinwang/Desktop/CS 221/cs221project"

class LstmClassifier(Model):
    def __init__(self,
            word_embeddings: TextFieldEmbedder,
            encoder: Seq2VecEncoder,
            vocab: Vocabulary) -> None:

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder

        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(), out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()

        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self,
            tokens: Dict[str, torch.Tensor],
            label: torch.Tensor=None) -> torch.Tensor:
        # pad shorter sequences with zeros to match lengths
        mask = get_text_field_mask(tokens)

        # forward pass
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.hidden2tag(encoder_out)
        output = {"logits": logits}

        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

@Predictor.register("sentence_classifier_predictor")
class SentenceClassifierPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        return self._dataset_reader.text_to_instance([Token(t) for t in tokens])

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

vocab = Vocabulary.from_files("{}/vocabulary".format(out_dir))
model = LstmClassifier(word_embeddings, lstm, vocab)
with open("{}/model.th".format(out_dir), 'rb') as f:
    model.load_state_dict(torch.load(f))
if cuda_device > -1:
    model.cuda(cuda_device)
predictor = SentenceClassifierPredictor(model, dataset_reader=reader)

# EVALUATION
true_pos = 0
false_pos = 0
false_neg = 0

fo = open('{}/test.txt'.format(out_dir), 'r')
lines = fo.readlines()
fo.close()
for line in lines:
    logits = predictor.predict(line[:-1])['logits']
    label_id = np.argmax(logits)

