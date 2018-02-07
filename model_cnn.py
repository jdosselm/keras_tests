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

from attention_layrer import Attention, AttentionWithContext

def get_config():
    return {
        'embedding_size': 300,
        'dropout_embedding': 0.6,

        'cnn_dilation_rates': '1',
        'cnn_windows': '12',
        'cnn_num_filters': 200,
        'cnn_filter_strides': '1',
        'cnn_pool_sizes': ['all,all,all'],

        'l2_reg_lambda': 0.001,

        'dense_layer_size': 20,
        'dropout_prob': 0.4
    }


def create_model(embeddings, config=get_config(), sentence_length=100):

    config['sentence_length']=sentence_length

    # sentence attention
    attention_input = Input(shape=(config['sentence_length'], config['embedding_size'],), dtype='float32')

    x = Permute((2, 1))(attention_input)
    x = Reshape((config['embedding_size'], config['sentence_length']))(x)
    x = Dense(config['sentence_length'], activation='softmax', bias=True)(x)

    x = Lambda(lambda x: K.mean(x, axis=1), name='attention_vector_sentence')(x)
    x = RepeatVector(config['embedding_size'])(x)
    # x = Lambda(lambda x: x, name='attention_vector_sentence')(x)

    attention_probabilities = Permute((2, 1))(x)

    x = multiply([attention_input, attention_probabilities], name='attention_mul')
    #x = Lambda(lambda x: K.sum(x, axis=1))(x)

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

    x_att = sentence_attention(x)
    x = BatchNormalization()(x)

    x_att_sum = Lambda(lambda x: K.sum(x, axis=1))(x_att)

    conv_results = []

    for k, d in enumerate(config['cnn_dilation_rates'].split(',')):

        for i, w in enumerate(config['cnn_windows'][k].split(',')):

            strides = int(config['cnn_filter_strides'].split(',')[i])

            window_size = int(w)

            if window_size <= config['sentence_length'] / int(d):

                for j, p in enumerate(config['cnn_pool_sizes'][k].split(',')):
                    if 'all' == p:
                        pool_size = config['sentence_length'] / int(d) - window_size + 1
                    else:
                        pool_size = int(p)

                    if pool_size > config['sentence_length'] / int(d) - window_size + 1:
                        pool_size = config['sentence_length'] / int(d) - window_size + 1

                    conv = Conv1D(
                        config['cnn_num_filters'],
                        window_size,
                        activation='relu',
                        use_bias=True,
                        kernel_regularizer=regularizers.l2(config['l2_reg_lambda']),
                        dilation_rate=int(d),
                        strides=strides
                    )

                    conv_results.append(conv(x_att))
                    conv_results[-1] = MaxPooling1D(pool_size, padding='valid')(conv_results[-1])
                    conv_results[-1] = Flatten()(conv_results[-1])
                    conv_results[-1] = ActivityRegularization(l2=config['l2_reg_lambda'])(conv_results[-1])

            else:
                raise Exception("Window too big .. exiting...")

    conv_results.append(x_att_sum)
    if len(conv_results) > 1:
        x = concatenate(conv_results, axis=1)
    else:
        x = conv_results[0]

    if config['dense_layer_size']:
        x = Dense(config['dense_layer_size'], activation='relu',
                  kernel_regularizer=regularizers.l2(config['l2_reg_lambda']))(x)
        x = Dropout(config['dropout_prob'])(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)

    return model, config


