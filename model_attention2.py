from keras.layers import Dense, Bidirectional, Input, Flatten, Embedding, concatenate, add
from tensorflow.contrib.keras.python.keras.layers import merge, add

from keras import backend as K
from keras.layers.core import SpatialDropout1D, Dropout, Reshape, Lambda, Permute, RepeatVector
from keras.models import Model

from keras import backend as K
from keras.layers import LSTM, Dense, Input, Embedding, Bidirectional, GRU
from keras.layers.core import SpatialDropout1D, Reshape, Lambda, Permute, RepeatVector
from keras.models import Model
from attention_layrer import Attention, AttentionWithContext

def get_config():
    return {
        'embedding_dropout': 0.7,
        'dense_layer': 20,
        'dropout_prob': 0.2,

        'lstm_layer_size': 300,
        'recurrent_dropout': 0.3,

    }


def create_model(embeddings, config=get_config(), sentence_length=100):

    config['sentence_length'] = sentence_length

    # sentence attention
    attention_input = Input(shape=(config['sentence_length'], 300,), dtype='float32')

    x = Permute((2, 1))(attention_input)
    x = Reshape((300, config['sentence_length']))(x)
    x = Dense(config['sentence_length'], activation='softmax', bias=True)(x)

    x = Lambda(lambda x: K.mean(x, axis=1), name='attention_vector_sentence')(x)
    x = RepeatVector(300)(x)
    # x = Lambda(lambda x: x, name='attention_vector_sentence')(x)

    attention_probabilities = Permute((2, 1))(x)

    x = merge.multiply([attention_input, attention_probabilities], name='attention_mul')
    x = Lambda(lambda x: K.sum(x, axis=1))(x)

    sentence_attention = Model(attention_input, x, name='sentence_attention')

    embedding_layer = Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=config['sentence_length'],
            trainable=False,
            weights=[embeddings],
        )

    input = Input(shape=(config['sentence_length'],), dtype='int32')
    x = embedding_layer(input)
    x = SpatialDropout1D(config['embedding_dropout'])(x)

    #x = Attention()(x)
    x1 = sentence_attention(x)
    #x2 = sentence_attention(x)
    x2 = GRU(config['lstm_layer_size'], return_sequences=False, recurrent_dropout=config['recurrent_dropout'], dropout=config['dropout_prob'])(x)

    x = add([x1, x2])

    if config['dense_layer']:
        x = Dense(config['dense_layer'], activation='relu')(x)
        x = Dropout(config['dropout_prob'])(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)

    return model, config




