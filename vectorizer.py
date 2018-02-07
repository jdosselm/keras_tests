from tf_keras.utils.vocabulary import Vocabulary


class VectorizerSpacy(object):
    def __init__(self, nlp=None):
        self.nlp = nlp
        self.unkown_index = 0

    def get_word_index(self, word):
        return self.nlp.vocab.vectors.find(key=word)

    def get_word_vector(self, word):
        return self.nlp.vocab.vectors.data[self.get_word_index(word)]


class VectorizerGlove(object):
    def __init__(self, dims=50):
        self.index2word, self.vocabulary_inv, self.w2v_matrix, self.unknown_word = Vocabulary(
            '/data/models/w2v/glove.6B.%sd.gensim' % dims).get_vocabulary_glove()
        self.unkown_index = self.vocabulary_inv.get(self.unknown_word)

    def get_word_index(self, word):
        return self.vocabulary_inv.get(word, self.unkown_index)

    def get_word_vector(self, word):
        return self.w2v_matrix[self.get_word_index(word)]

    def get_word_matrix(self):
        return self.w2v_matrix


