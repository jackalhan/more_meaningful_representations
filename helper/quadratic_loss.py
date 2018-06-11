import tensorflow as tf

def euclidean_distance_loss(embeddings, target_embeddings, params, is_training=True, type='old_loss'):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """

    difference = tf.subtract(embeddings, target_embeddings)
    if type == 'old_loss':
        euc_dist = tf.sqrt(tf.reduce_sum(tf.square(difference), axis=1))
        if is_training:
            subtracted_margin = tf.subtract(euc_dist, params.margin)
            added_margin = tf.reduce_mean(tf.nn.relu(subtracted_margin)) + params.margin
            loss = added_margin
        else:
            loss = euc_dist
    elif type == 'new_loss':
        euc_dist = tf.sqrt(tf.reduce_sum(tf.square(difference), axis=1))
        loss = tf.reduce_mean(tf.nn.relu(euc_dist - params.margin) + tf.multiply(tf.cast(euc_dist > params.margin, dtype=tf.float32), params.margin))
    else:
        raise ValueError("Loss strategy type is not recognized: {}".format(type))

    return loss