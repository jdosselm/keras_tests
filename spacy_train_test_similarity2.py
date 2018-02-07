import plac
import pathlib
import spacy
import numpy as np
import pickle
from keras.models import Model

from spacy_data import get_train_dev_for_embedding, get_embeddings

from model_similarity2 import create_model

from spacy_features import anygram_kernel, anygram_similarity_fast as kernel, words_to_indices as get_features
from vectorizer import VectorizerGlove as Vectorizer

print "Loading spacy..."
nlp = spacy.load('en_core_web_lg')

print "Loading vectorizer..."
vectorizer = Vectorizer(dims=50)


def train(train_texts, train_labels, dev_texts, dev_labels, sentence_length=100, batch_size=100, nb_epoch=5):

    #embeddings, vocab = get_embeddings(nlp.vocab)
    # from spacy_features import w2v_matrix
    embeddings = vectorizer.get_word_matrix()
    vocab = None

    print embeddings.shape

    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_texts))
    dev_docs = list(nlp.pipe(dev_texts))

    # get anygram sims to all docs, as output of the model
    anygram_file = 'anygrams.np'
    try:
        train_labels = np.load(anygram_file + '-train.npy')
        dev_labels = np.load(anygram_file + '-dev.npy')
        print "loaded anygram matrix.."
    except:
        print("Calculating anygram sims...")
        train_labels = anygram_kernel(train_docs, train_docs, kernel=kernel, vectorizer=vectorizer)
        np.save(anygram_file + '-train', train_labels2)

        dev_labels = anygram_kernel(dev_docs, train_docs, kernel=kernel, vectorizer=vectorizer)
        np.save(anygram_file + '-dev', dev_labels2)

    # print a.shape
    # train_labels2 = a[:len(train_docs)]
    # dev_labels2 = a[len(train_docs):]

    print("Creating model...")
    model, config = create_model(embeddings, sentence_length=sentence_length, output_length=len(train_labels))

    model.compile(
        optimizer='sgd',
        loss='mean_squared_error',
        loss_weights=[1.],
        metrics=['accuracy']
    )

    print(model.summary())

    print("Creating input...")
    train_X = get_features(train_docs, vocab, sentence_length, vectorizer=vectorizer)
    dev_X = get_features(dev_docs, vocab, sentence_length, vectorizer=vectorizer)

    print train_X[:2]
    # print train_X['input_2'][:2]
    print train_labels[:2]
    # print dev_X['input_1'][:2]
    # print dev_X['input_2'][:2]
    # print dev_labels[:2]

    # print("Class weights...")
    # from sklearn.utils import class_weight
    # class_weight = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
    # class_weight = dict(zip(xrange(len(class_weight)), class_weight))
    # print class_weight

    print("Train...")
    model.fit(train_X,
              train_labels,
              validation_data=(dev_X, dev_labels),
              epochs=nb_epoch,
              batch_size=batch_size,
              # class_weight=class_weight
              )

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

    Xs = get_features(nlp, texts, vocab, sentence_length, vectorizer=vectorizer)

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


def load_model(model_path):
    from keras.models import model_from_json
    import pickle
    from spacy_data import get_embeddings
    from attention_layer import Attention, AttentionWithContext

    with (model_path / 'config.json').open() as file_:
        model = model_from_json(file_.read(),
                                custom_objects={'Attention': Attention, 'AttentionWithContext': AttentionWithContext})

    with (model_path / 'model').open('rb') as file_:
        weights = pickle.load(file_)

    # embeddings, vocab = get_embeddings(nlp.vocab)
    # from spacy_features import w2v_matrix
    # embeddings = w2v_matrix
    # vocab = None

    embeddings = vectorizer.get_word_matrix()
    vocab = None

    model.set_weights([embeddings] + weights)
    return model, vocab


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
        sentence_length=24,
        nb_epoch=10,
        batch_size=20
    ):

    if model_dir is not None:
        model_dir = pathlib.Path(model_dir)

    print "Loading data..."

    test = ['CBP Group']
    # test=['Twitter'],
    # test=['Aera set1 2017-11-21'],
    # test=['SSGA'],
    # test=['x'],
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = get_train_dev_for_embedding(test=test)

    if not eval:

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

    else:

        model, vocab = load_model(model_dir)

        layer_name = 'gru_1'
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        test_docs = list(nlp.pipe(train_texts))

        Xs = get_features(test_docs, vocab, sentence_length, vectorizer=vectorizer)

        intermediate_output = intermediate_layer_model.predict(Xs)

        # print intermediate_output

        from sklearn.manifold import TSNE
        from sklearn.metrics.pairwise import pairwise_distances

        pw = pairwise_distances(intermediate_output, metric='cosine')
        embedded_word_vectors = TSNE(n_components=2, n_iter=500, random_state=0, metric='precomputed').fit_transform(pw)
        print embedded_word_vectors
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        color = ['green' if l else 'blue' for l in test_labels]

        plt.scatter(embedded_word_vectors[:,0], embedded_word_vectors[:,1], color=color)
        plt.savefig('/tmp/f.png' )



        # print "Evaluating..."
        # acc, prec, rec, f1 = evaluate(model_dir, test_texts, test_labels, sentence_length=sentence_length)
        # print "-" * 100
        # print test
        # print "-" * 100
        # print "Accuracy: %s   F1: %s   Precision: %s   Recall: %s" % (acc, f1, prec, rec)
        # print "-" * 100


if __name__ == '__main__':
    plac.call(main)
