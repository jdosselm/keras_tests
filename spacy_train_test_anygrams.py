import plac
import pathlib
import spacy
import numpy as np
import pickle

from spacy_data import get_train_dev_test, get_embeddings

from model_similarity2 import create_model

from spacy_features import get_hv_similar_sentences as get_features

print "Loading spacy..."
nlp = spacy.load('en_core_web_lg')


def train(train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels, sentence_length=100, batch_size=100, nb_epoch=5):

    embeddings, vocab = get_embeddings(nlp.vocab)

    print embeddings.shape

    print("Creating model...")
    from sklearn import svm
    from spacy_features import anygram_kernel

    # 0.2 -- 0.3 0.754
    # 1.0 -- 0.1 0.75

    for c, l in [(0.8, 0.8)]:
    # for c in [0.1, 0.2, 0.3, 0.4, 0.8, 1.0]:
    #     for l in [0.1, 0.3, 0.5, 0.8, 1.0]:
            C = c
            clf = svm.SVC(C=C, kernel='precomputed', class_weight='balanced', verbose=True)

            print("Parsing texts...")
            train_docs = list(nlp.pipe(train_texts))
            dev_docs = list(nlp.pipe(dev_texts))
            test_docs = list(nlp.pipe(test_texts))

            print("Creating input...")
            train_X = anygram_kernel(train_docs, train_docs, l=l)
            print train_X.shape

            # train_X = get_features(nlp, train_texts, vocab, sentence_length)
            # dev_X = get_features(nlp, dev_texts, vocab, sentence_length)

            print("Class weights...")
            from sklearn.utils import class_weight
            class_weight = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
            class_weight = dict(zip(xrange(len(class_weight)), class_weight))
            print class_weight

            train_labels = [int(b) for b in train_labels]
            dev_labels = [int(b) for b in dev_labels]
            test_labels = [int(b) for b in test_labels]

            print("Train...")
            model = clf.fit(train_X, train_labels)

            print("Test...")
            # p = model.predict(anygram_kernel(dev_docs[:10], train_docs))
            # print p

            s = model.score(anygram_kernel(test_docs, train_docs), test_labels)
            #s = model.score(anygram_kernel(dev_docs, train_docs, l=l), dev_labels)

            print c, l, s

    return model


def evaluate(model_path, texts, labels, sentence_length=100):
    from keras.models import model_from_json
    import pickle
    from spacy_data import get_embeddings
    from attention_layer import Attention, AttentionWithContext

    with (model_path / 'config.json').open() as file_:
        model = model_from_json(file_.read(),
                                custom_objects={'Attention': Attention, 'AttentionWithContext': AttentionWithContext})

    with (model_path / 'model').open('rb') as file_:
        weights = pickle.load(file_)

    embeddings, vocab = get_embeddings(nlp.vocab)
    model.set_weights([embeddings] + weights)

    Xs = get_features(nlp, texts, vocab, sentence_length)

    acc = precision = recall = f1 = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    t = 0.5
    ys = model.predict(Xs)
    for i, y in enumerate(ys):
        # print y, labels[i]
        claim_score = y[0]
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

        #test = ['CBP Group']
        #test=['Twitter'],
        #test=['Aera set1 2017-11-21']
        #test=['SSGA']
        #test=['x'],

        for test in [['CBP Group'], ['Aera set1 2017-11-21'], ['SSGA']]:
            train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = get_train_dev_test(test=test, split=0.8)

            print "+" * 20
            print test

            model = train(
                train_texts, train_labels,
                dev_texts, dev_labels,
                test_texts, test_labels,
                sentence_length=sentence_length,
                nb_epoch=nb_epoch,
                batch_size=batch_size
            )
            print "+" * 20

        # if model_dir:
        #     print "Storing model.."
        #     weights = model.get_weights()
        #
        #     with (model_dir / 'model').open('wb') as file_:
        #         pickle.dump(weights[1:], file_)
        #
        #     with (model_dir / 'config.json').open('wb') as file_:
        #         file_.write(model.to_json())


    # print "Evaluating..."
    # acc, prec, rec, f1 = evaluate(model_dir, test_texts, test_labels, sentence_length=sentence_length)
    # print "-" * 100
    # print test
    # print "-" * 100
    # print "Accuracy: %s   F1: %s   Precision: %s   Recall: %s" % (acc, f1, prec, rec)
    # print "-" * 100


if __name__ == '__main__':
    plac.call(main)
