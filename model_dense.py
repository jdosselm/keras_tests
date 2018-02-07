from keras.layers import Dense, Bidirectional, Input, Flatten, Embedding, TimeDistributed
from tensorflow.contrib.keras.python.keras.layers import merge, add

from keras import backend as K
from keras.layers.core import SpatialDropout1D, Dropout, Reshape, Lambda, Permute, RepeatVector
from keras.models import Model
from attention_layrer import Attention, AttentionWithContext

def get_config():
    return {
        'embedding_dropout': 0.6,
        'dense_layer': 20,
        'dropout_prob': 0.1
    }


def create_model(embeddings, config=get_config(), sentence_length=100):

    config['sentence_length'] = sentence_length

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

    x = Lambda(lambda x: K.sum(x, axis=1))(x)

    if config['dense_layer']:
        x = Dense(config['dense_layer'], activation='sigmoid')(x)
        x = Dropout(config['dropout_prob'])(x)

    output = Dense(1, activation='sigmoid',)(x)

    model = Model(inputs=input, outputs=output)

    return model, config




