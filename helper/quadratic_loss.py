import tensorflow as tf

def euclidean_distance_loss(embeddings, target_embeddings, params, is_training=True, type='square'):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    with tf.variable_scope('euclidean_distance'):
        difference = tf.subtract(embeddings, target_embeddings)
        if type == 'square':
            loss = tf.reduce_mean(tf.square(difference))
        else:
            loss = tf.sqrt(tf.reduce_sum(tf.square(difference), axis=1))
            if is_training:
                loss = tf.subtract(loss, params.margin)
                loss = tf.reduce_mean(tf.nn.relu(loss)) + params.margin
        return loss