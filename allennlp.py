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

torch.manual_seed(1)

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
        
config = Config(
    testing=True,
    seed=1,
    batch_size=64,
    lr=3e-4,
    epochs=2,
    hidden_sz=64,
    max_seq_len=100, # necessary to limit memory usage
    max_vocab_size=100000,
)

class EMRDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
 
    @overrides
    def text_to_instance(self, tokens: List[Token], label: str = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        if label:
            label_field = LabelField(label=label)
            fields["label"] = label_field

        return Instance(fields)
     
    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                words = line.strip().split()
                yield self.text_to_instance([Token(word) for word in words[:-1]], words[-1])

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

reader = EMRDatasetReader()

train_dataset = reader.read('{}/train.txt'.format(out_dir))
val_dataset = reader.read('{}/val.txt'.format(out_dir))

#print(vars(train_dataset[0].fields["tokens"]))

vocab = Vocabulary.from_instances(train_dataset + val_dataset)
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

# LSTM-RNN implementation as encoder
lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = LstmClassifier(word_embeddings, lstm, vocab)

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
iterator.index_with(vocab)

# tune hyperparameters
trainer = Trainer(model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        patience=10,
        num_epochs=20,
        cuda_device=cuda_device)

trainer.train()

predictor = SentenceClassifierPredictor(model, dataset_reader=reader)

# SANITY CHECK
logits = predictor.predict("urinari pouch stone stent placement")['logits']
label_id = np.argmax(logits)
print(model.vocab.get_token_from_index(label_id, 'labels'))

# save the model
with open("{}/model.th".format(out_dir), 'wb') as f:
    torch.save(model.state_dict(), f)
vocab.save_to_files("{}/vocabulary".format(out_dir))

# reload the model
vocab2 = Vocabulary.from_files("{}/vocabulary".format(out_dir))
model2 = LstmClassifier(word_embeddings, lstm, vocab2)
with open("{}/model.th".format(out_dir), 'rb') as f:
    model2.load_state_dict(torch.load(f))
if cuda_device > -1:
    model2.cuda(cuda_device)
predictor2 = SentenceClassifierPredictor(model2, dataset_reader=reader)
logits2 = predictor2.predict("urinari pouch stone stent placement")['logits']
np.testing.assert_array_almost_equal(logits2, logits)