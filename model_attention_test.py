from keras.layers import Dense, Bidirectional, Input, Flatten, Embedding, TimeDistributed
from tensorflow.contrib.keras.python.keras.layers import merge, add, concatenate, average

from keras import backend as K
from keras.layers.core import SpatialDropout1D, Dropout, Reshape, Lambda, Permute, RepeatVector
from keras.models import Model
from attention_layrer import Attention, AttentionWithContext

def get_config():
    return {
        'embedding_size': 3,
        'embedding_dropout': 0.65,
        'dense_layer': None,
        'dropout_prob': 0.2
    }


def create_model(embeddings, config=get_config(), sentence_length=100):

    config['sentence_length'] = sentence_length

    # sentence attention
    attention_input = Input(shape=(config['sentence_length'], config['embedding_size'],), dtype='float32')

    x = Permute((2, 1))(attention_input)
    x = Reshape((config['embedding_size'], config['sentence_length']))(x)
    x = Dense(config['sentence_length'], activation='softmax', bias=True)(x)

    x = Lambda(lambda x: K.mean(x, axis=1), name='attention_vector_sentence')(x)
    x = RepeatVector(config['embedding_size'])(x)
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

    pos_mbedding_layer = Embedding(
            15,
            config['embedding_size'],
            input_length=config['sentence_length'],
            trainable=True,
            # weights=[embeddings],
    )

    input = Input(shape=(config['sentence_length'],), dtype='int32', name='input_1')
    x = embedding_layer(input)
    x = SpatialDropout1D(config['embedding_dropout'])(x)
    x1 = Attention()(x)

    input2 = Input(shape=(config['sentence_length'],), dtype='int32', name='input_2')
    x2 = pos_mbedding_layer(input2)
    x2 = sentence_attention(x2)

    # res
    x_sum = Lambda(lambda x: K.mean(x, axis=1))(x)
    x1 = add([x1, x_sum])

    x = concatenate([x1, x2])

    if config['dense_layer']:
        x = Dense(config['dense_layer'], activation='relu')(x)
        x = Dropout(config['dropout_prob'])(x)

    output = Dense(1, activation='sigmoid',)(x)

    model = Model(inputs=[input, input2], outputs=output)

    return model, config




