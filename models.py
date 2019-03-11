import tensorflow as tf
# or, import keras

def embedding_layer(ids_, V, embed_dim, init_scale=0.001):
    """Construct an embedding layer.

    Initialization options include:
        - word2vec
        - GloVe
        - random_uniform_initializer
        - others???

    You should define a variable for the embedding matrix, and initialize it
    using tf.random_uniform_initializer to values in [-init_scale, init_scale].
    Hint: use tf.nn.embedding_lookup
    Args:
        ids_: [batch_size, max_len] Tensor of int32, integer ids
        V: (int) vocabulary size
        embed_dim: (int) embedding dimension
        init_scale: (float) scale to initialize embeddings
    Returns:
        xs_: [batch_size, max_len, embed_dim] Tensor of float32, embeddings for
            each element in ids_
    """

    W_embed_ = tf.get_variable("W_embed", shape=[V, embed_dim],
                        initializer=tf.random_uniform_initializer(minval=-init_scale, maxval=init_scale))


    xs_ = tf.nn.embedding_lookup(params = W_embed_, ids = ids_)

    return xs_

def feed_forward_layers():

    """This will be just like what I did in A3."""
    pass

def softmax_output_layer():
    """Again, just like A3"""
    pass

def CRF_output_layer_():
    """New wrinkle: in addition to softmax output, we will try CRF"""


def CNN_layers():
    pass

def LSTM_layers():
    pass

def fully_formed_model():
    """ will take args about the embedding (what kind), the base model (cnn vs lstm), what their hyperparameters will be
    and what type of output layer"""
    pass
