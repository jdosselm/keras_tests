import numpy as np
from tree_utils import get_flattened
from spacy.tokens import Doc
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

# print "Loading glove..."
# from tf_keras.utils.vocabulary import Vocabulary
# index2word, vocabulary_inv, w2v_matrix, unknown_word = Vocabulary('/data/models/w2v/glove.6B.50d.gensim').get_vocabulary_glove()
# print "loaded"


pos = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'POS', 'PRON', 'PROPN', 'PUNCT', 'SPACE', 'SYM', 'VERB', 'X']


def get_word_vector_spacy(token):
    return token.vector


# def get_word_vector_glove(token):
#     return w2v_matrix[vocabulary_inv.get(token.lower_, vocabulary_inv.get(unknown_word))]


def words_to_indices(sentences, vocabulary, max_length, lemmas=False, vectorizer=None):
    sentences = list(sentences)
    Xs = np.zeros((len(sentences), max_length), dtype='int32')
    for i, sent in enumerate(sentences):
        j = 0
        for token in sent:
            if lemmas:
                word = token.lemma
            else:
                word = token.lower_
            vector_id = vectorizer.get_word_index(word)
            if vector_id >= 0:
                Xs[i, j] = vector_id
            else:
                Xs[i, j] = 0
            j += 1
            if j >= max_length:
                break
    return Xs

#
# def words_to_indices_glove(sentences, vocabulary, max_length, lemmas=False):
#     sentences = list(sentences)
#     Xs = np.zeros((len(sentences), max_length), dtype='int32')
#     for i, sent in enumerate(sentences):
#         j = 0
#         for token in sent:
#             if lemmas:
#                 word = token.lemma
#             else:
#                 word = token.lower_
#             vector_id = vocabulary_inv.get(word, vocabulary_inv.get(unknown_word))
#             if vector_id >= 0:
#                 Xs[i, j] = vector_id
#             else:
#                 Xs[i, j] = 0
#             j += 1
#             if j >= max_length:
#                 break
#     return Xs


def sentences_to_trees(sentences, vocabulary, max_length, lemmas=False):
    vocab = sentences[0][0].vocab
    flattened = list()
    for s in sentences:
        f = get_flattened(s).split()
        flattened.append(Doc(vocab, words=f))
    return words_to_indices(flattened, vocabulary, max_length, lemmas=lemmas)


def sentence_to_pos(sentences, vocabulary, max_length, lemmas=False):
    Xs = np.zeros((len(sentences), max_length), dtype='int32')
    for i, s in enumerate(sentences):
        for j, t in enumerate(s):

            if j >= max_length:
                break

            try:
                Xs[i, j] = pos.index(t.pos_)
            except:
                Xs[i, j] = pos.index('X')

    return Xs


def sequencial_and_trees(sentences, vocabulary, max_length, lemmas=False):
    return {
        'input_1': words_to_indices(sentences, vocabulary, max_length, lemmas=False),
        'input_2': sentence_to_pos(sentences, vocabulary, max_length, lemmas=False)
    }


def get_hv_similar_sentences(nlp, sentence_tuples, vocabulary, max_length, lemmas=False):

    s1 = list()
    s2 = list()

    for t in sentence_tuples:
        s1.append(t[0])
        s2.append(t[1])

    print len(s1)
    s1d = list(nlp.pipe(s1, n_threads=4, disable=['parser', 'ner', 'tagger']))
    print len(s2)
    s2d = list(nlp.pipe(s2, n_threads=4, disable=['parser', 'ner', 'tagger']))

    s1d = words_to_indices(s1d, vocabulary, max_length, lemmas=lemmas)
    s2d = words_to_indices(s2d, vocabulary, max_length, lemmas=lemmas)

    result = [s1d, s2d]
    # result = list()
    # for i, s in enumerate(s1d):
    #     result.append((s, s2d[i]))

    return result

    #
    # return {
    #     'input_1': words_to_indices(s1d, vocabulary, max_length, lemmas=lemmas),
    #     'input_2': words_to_indices(s2d, vocabulary, max_length, lemmas=lemmas),
    # }


def anygram_similarity(s1, s2, l=0.8, threshold=0.7):
    kernel = 0
    s1l = len(s1)
    s2l = len(s2)

    # init w/ zeros
    delta = np.zeros((s1l + 1, s2l + 1), dtype='float32')
    # cosine sim per word pair
    c = cosine_similarity([t.vector for t in s1], [t.vector for t in s2])
    # cut off low sims ?
    c[np.where((c < threshold) & (c > -threshold))] = 0

    # iterate starting last row
    for i in xrange(s1l - 1, -1, -1):
        for j in xrange(0, s2l):
            delta[i][j] = l * (c[i][j] + delta[i+1][j+1])
            kernel += delta[i][j]
    return kernel


def anygram_similarity_fast(s1, s2, l=0.8, delta_factor=1.0, normalize=False, stopword_factor=1.0, threshold=0.0, vectorizer=None):
    # word pair sims
    c = cosine_similarity([vectorizer.get_word_vector(t.lower_) for t in s1], [vectorizer.get_word_vector(t.lower_) for t in s2])
    # cut off below x ?
    c[np.where((c < threshold) & (c > -threshold))] = 0
    # init w/ zeros
    c2 = np.zeros((c.shape[0] + 1, c.shape[1] + 1))
    # expanded c w/ last row / col = 0
    c2[:-1, :-1] = c

    # moving square
    for i in range(min(c.shape[0], c.shape[1])):
        # last row / col = previous last result
        c3 = np.zeros((c2.shape[0], c2.shape[1]))
        c3[:-i - 1, :-i - 1] = delta_factor * (c2[1:-i, 1:-i] if i > 0 else c2[1:, 1:])  # c2[i+1:, i+1:]
        c3[:-i - 2, :-i - 2] = 0

        # add to uderlying sims
        c2 = np.add(c2, c3)

        # decay factor
        c2[-i-1-1:-i-1, :-i-1-1] *= l
        c2[:-i-1, -i-1-1:-i-1] *= l

    return np.sum(c2)


def anygram_kernel(X, Y, kernel=anygram_similarity_fast, vectorizer=None):
    ht = dict()
    result = np.zeros((len(X), len(Y)), dtype='float32')
    for i in xrange(0, len(X)):

        for j in xrange(0, len(Y)):
            h = hash(X[i].text) + hash(Y[j].text)
            if ht.get(h):
                result[i, j] = ht.get(h)
            else:
                result[i, j] = kernel(X[i], Y[j], vectorizer=vectorizer)
                ht[h] = result[i, j]

    # scale ?
    # r = r / r.max(axis=1)[:, None]
    # r = np.transpose(np.triu(r, 0)) + np.triu(r, 1)
    return result


def anygram_similarity_dep(s1, s2, l=1.0):
    kernel = 0
    s1l = len(s1)
    s2l = len(s2)
    delta = np.zeros((s1l + 1, s2l + 1), dtype='float32')
    for i in xrange(s1l - 1, -1, -1):
        for j in xrange(0, s2l):
            delta[i][j] = l * ((1 if s1[i].dep_ == s2[j].dep_ else 0) + delta[i + 1][j + 1])
            kernel += delta[i][j]
    return kernel


def anygram_similarity_pos(s1, s2, l=1.0):
    kernel = 0
    s1l = len(s1)
    s2l = len(s2)
    delta = np.zeros((s1l + 1, s2l + 1), dtype='float32')
    for i in xrange(s1l - 1, -1, -1):
        for j in xrange(0, s2l):
            delta[i][j] = l * ((1 if s1[i].pos_ == s2[j].pos_ else 0) + delta[i + 1][j + 1])
            kernel += delta[i][j]
    return kernel

