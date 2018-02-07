import plac
import pathlib
import spacy
import numpy as np
import pickle

from spacy_pipe import KerasPipe
from spacy_data import get_train_dev_test, get_embeddings

from model_lstm import create_model

#from model_cnn import create_model
#from model_cnn_rnn import create_model
from model_attention import create_model
#from model_attention_test import create_model
#from model_attention2 import create_model
#from model_dense import create_model

from spacy_features import words_to_indices as get_features
#from spacy_features import sentences_to_trees as get_features
#from spacy_features import sequencial_and_trees as get_features

print "Loading spacy..."
nlp = spacy.load('en_core_web_lg')

# with open('/data/models/w2v/glove.6B.50d.gensim', 'rb') as file_:
#     header = file_.readline()
#     nr_row, nr_dim = header.split()
#     nlp.vocab.reset_vectors(width=int(nr_dim))
#     for line in file_:
#         line = line.rstrip().decode('utf8')
#         pieces = line.rsplit(' ', int(nr_dim))
#         word = pieces[0]
#         vector = np.asarray([float(v) for v in pieces[1:]], dtype='f')
#         nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab


def train(train_texts, train_labels, dev_texts, dev_labels, sentence_length=100, batch_size=100, nb_epoch=5):

    embeddings, vocab = get_embeddings(nlp.vocab)

    print embeddings.shape

    print("Creating model...")
    model, config = create_model(embeddings, sentence_length=sentence_length)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        loss_weights=[1.],
        metrics=['accuracy']
    )

    print(model.summary())

    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_texts))
    dev_docs = list(nlp.pipe(dev_texts))

    print("Creating input...")
    train_X = get_features(train_docs, vocab, sentence_length)
    dev_X = get_features(dev_docs, vocab, sentence_length)

    print("Class weights...")
    from sklearn.utils import class_weight
    class_weight = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
    class_weight = dict(zip(xrange(len(class_weight)), class_weight))
    print class_weight

    print("Train...")
    model.fit(train_X,
              train_labels,
              validation_data=(dev_X, dev_labels),
              epochs=nb_epoch,
              batch_size=batch_size,
              class_weight=class_weight
              )

    return model


def evaluate(model_dir, texts, labels, sentence_length=100):
    s = KerasPipe.load(model_dir, nlp, sentence_length, get_features)
    nlp.add_pipe(s, last=True)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    t = 0.5
    for i, doc in enumerate(nlp.pipe(texts, batch_size=1000, n_threads=4)):
        claim_score = list(doc.sents)[0]._.claim_score
        tp += claim_score >= t and bool(labels[i]) == True
        tn += claim_score < t and bool(labels[i]) == False
        fp += claim_score >= t and bool(labels[i]) == False
        fn += claim_score < t and bool(labels[i]) == True

    acc = float(tp + tn) / (tp+tn+fp+fn)
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1 = float(2*tp) / (2 * tp + fp + fn)

    return acc, precision, recall, f1


@plac.annotations(
    model_dir=("directory of (to be) stored model",),
    eval=("evaluate test data", "flag", "e", bool),
    sentence_length=("Maximum sentence length", "option", "L", int),
    nb_epoch=("Number of training epochs", "option", "i", int),
    batch_size=("Size of mini-batches for training", "option", "b", int)
)
def main(
        model_dir='spacy_keras_model',
        eval=False,
        sentence_length=40,
        nb_epoch=10,
        batch_size=20
    ):

    if model_dir is not None:
        model_dir = pathlib.Path(model_dir)

    if not eval:
        print "Loading data..."

        test = ['CBP Group']
        #test=['Twitter'],
        #test=['Aera set1 2017-11-21'],
        #test=['SSGA'],
        #test=['x'],

        train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = get_train_dev_test(test=test)

        model = train(
            train_texts, train_labels,
            dev_texts, dev_labels,
            sentence_length=sentence_length,
            nb_epoch=nb_epoch,
            batch_size=batch_size
        )

        if model_dir:
            print "Storing model.."
            weights = model.get_weights()

            with (model_dir / 'model').open('wb') as file_:
                pickle.dump(weights[1:], file_)

            with (model_dir / 'config.json').open('wb') as file_:
                file_.write(model.to_json())

    print "Evaluating..."
    acc, prec, rec, f1 = evaluate(model_dir, test_texts, test_labels, sentence_length=sentence_length)
    print "-" * 100
    print test
    print "-" * 100
    print "Accuracy: %s   F1: %s   Precision: %s   Recall: %s" % (acc, f1, prec, rec)
    print "-" * 100


if __name__ == '__main__':
    plac.call(main)
