from spacy.tokens import Span
from keras.models import model_from_json
import pickle
import cytoolz

from spacy_data import get_embeddings
from attention_layer import Attention, AttentionWithContext


class KerasPipe(object):
    extension_name = 'claim_score'

    """
    to be used as a spacy pipeline dropin:
    p = KerasPipe.load(model_dir, nlp)
    nlp.add_pipe(p, last=True)
    """
    @classmethod
    def load(cls, model_path, nlp, sentence_length, get_features):
        with (model_path / 'config.json').open() as file_:
            model = model_from_json(file_.read(), custom_objects={'Attention': Attention, 'AttentionWithContext': AttentionWithContext})

        with (model_path / 'model').open('rb') as file_:
            weights = pickle.load(file_)

        embeddings, vocab = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + weights)

        return cls(model, vocab, sentence_length, get_features)

    def __init__(self, model, vocab, sentence_length, get_features):
        self._model = model
        self._vocab = vocab
        self._get_features = get_features
        self.sentence_length = sentence_length
        Span.set_extension(KerasPipe.extension_name, default=0.0)

    def __call__(self, doc):
        X = self.get_features(doc.sents, self._vocab, self.sentence_length)
        y = self._model.predict(X)

        for sent, label in zip(doc.sents, y):
            sent._.set(KerasPipe.extension_name, label[0])

        return doc

    def pipe(self, docs, batch_size=1000, n_threads=0):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)

            sentences = list()
            for doc in minibatch:
                sentences.extend(doc.sents)

            Xs = self.get_features(sentences, self._vocab, self.sentence_length)
            ys = self._model.predict(Xs)

            for sent, label in zip(sentences, ys):
                sent._.set(KerasPipe.extension_name, label[0])

            for doc in minibatch:
                yield doc

    def get_features(self, sentences, vocab, sentence_length):
        """
        convert sentences into network input
        """
        return self._get_features(sentences, vocab, sentence_length)

