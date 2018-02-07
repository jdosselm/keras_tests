from keras.layers import Dense, Embedding, Input, SpatialDropout1D, Flatten, Dropout, ActivityRegularization, concatenate, dot
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.models import Model
from tensorflow.contrib.keras import regularizers

from keras.layers import Dense, Bidirectional, Input, Flatten, Embedding, TimeDistributed, multiply

from keras import backend as K
from keras.layers.core import SpatialDropout1D, Dropout, Reshape, Lambda, Permute, RepeatVector
from keras.models import Model
from keras.layers.normalization import BatchNormalization

from keras import backend as K
from keras.layers import LSTM, Dense, Input, Embedding, Bidirectional, GRU
from keras.layers.core import SpatialDropout1D, Reshape, Lambda, Permute, RepeatVector
from keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import merge

from attention_layrer import Attention, AttentionWithContext

def get_config():
    return {
        'embedding_size': 50,
        'embedding_dropout': 0.6,

        'cnn_dilation_rates': '1',
        'cnn_windows': '12',
        'cnn_num_filters': 200,
        'cnn_filter_strides': '1',
        'cnn_pool_sizes': ['all,all,all'],

        'lstm_layer_size': 32,
        'recurrent_dropout': 0.2,

        'l2_reg_lambda': 0.001,

        'attention': False,

        'dense_layer_size': None,
        'dropout_prob': 0.2
    }


def create_model(embeddings, config=get_config(), sentence_length=100, output_length=160):

    config['sentence_length'] = sentence_length

    embedding_layer = Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=config['sentence_length'],
            trainable=False,
            weights=[embeddings],
        )

    input = Input(shape=(config['sentence_length'],), dtype='int32', name='input')

    x = embedding_layer(input)
    x = SpatialDropout1D(config['embedding_dropout'])(x)
    #x = Bidirectional(GRU(config['lstm_layer_size'], return_sequences=config['attention'], recurrent_dropout=config['recurrent_dropout'], dropout=config['dropout_prob']))(x)
    x = GRU(config['lstm_layer_size'], return_sequences=config['attention'], recurrent_dropout=config['recurrent_dropout'], dropout=config['dropout_prob'])(x)

    if config['dense_layer_size']:
        x = Dense(config['dense_layer_size'])(x)

    output = Dense(output_length, activation='linear')(x)

    # output2 = Dense(output_length, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)

    return model, config

