from keras.layers import Dense, Embedding, Input, SpatialDropout1D, Flatten, Dropout, ActivityRegularization, concatenate
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
        'embedding_size': 30,
        'dropout_embedding': 0.6,

        'cnn_dilation_rates': '1',
        'cnn_windows': '12',
        'cnn_num_filters': 200,
        'cnn_filter_strides': '1',
        'cnn_pool_sizes': ['all,all,all'],

        'lstm_layer_size': 32,
        'recurrent_dropout': 0.2,

        'l2_reg_lambda': 0.001,

        'attention': True,

        'dense_layer_size': None,
        'dropout_prob': 0.2
    }


def create_model(embeddings, config=get_config(), sentence_length=100):

    config['sentence_length']=sentence_length

    # sentence attention
    attention_input = Input(shape=(config['sentence_length'] - 2, config['embedding_size'],), dtype='float32')

    x = Permute((2, 1))(attention_input)
    x = Reshape((config['embedding_size'], config['sentence_length'] - 2))(x)
    x = Dense(config['sentence_length'] - 2, activation='softmax', bias=True)(x)

    x = Lambda(lambda x: K.mean(x, axis=1), name='attention_vector_sentence')(x)
    x = RepeatVector(config['embedding_size'])(x)
    # x = Lambda(lambda x: x, name='attention_vector_sentence')(x)

    attention_probabilities = Permute((2, 1))(x)

    x = multiply([attention_input, attention_probabilities], name='attention_mul')
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
    x = SpatialDropout1D(config['dropout_embedding'])(x)

    x = Conv1D(
        config['cnn_num_filters'],
        3,
        # activation='relu',
        use_bias=True,
        # kernel_regularizer=regularizers.l2(config['l2_reg_lambda']),
        # strides=1
    )(x)


    # x = MaxPooling1D(1, padding='valid')(x)
    # x = Flatten()(x)
    #x = Attention()(x)

    #x = Bidirectional(GRU(config['lstm_layer_size'], return_sequences=config['attention'], recurrent_dropout=config['recurrent_dropout'], dropout=config['dropout_prob']))(x)
    # x = GRU(config['lstm_layer_size'], return_sequences=config['attention'], recurrent_dropout=config['recurrent_dropout'], dropout=config['dropout_prob'])(x)

    if config['attention']:
        #x = sentence_attention(x)
        x = Attention()(x)

    # x = ActivityRegularization(l2=config['l2_reg_lambda'])(x)

    #
    # conv_results[-1] = ActivityRegularization(l2=config['l2_reg_lambda'])(conv_results[-1])

    if config['dense_layer_size']:
        x = Dense(config['dense_layer_size'], activation='relu',
                  kernel_regularizer=regularizers.l2(config['l2_reg_lambda']))(x)
        x = Dropout(config['dropout_prob'])(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)

    return model, config


