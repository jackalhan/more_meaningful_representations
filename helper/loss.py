import tensorflow as tf

def euclidean_distance_loss(question_embeddings, paragraph_embeddings, params):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    difference = tf.subtract(question_embeddings, paragraph_embeddings)
    if params.loss["version"] == 1:
        euc_dist = tf.sqrt(tf.reduce_sum(tf.square(difference), axis=1))
        subtracted_margin = tf.subtract(euc_dist, params.loss["margin"])
        added_margin = tf.reduce_mean(tf.nn.relu(subtracted_margin)) + params.loss["margin"]
        loss = added_margin
    elif params.loss["version"] == 2:
        euc_dist = tf.sqrt(tf.reduce_sum(tf.square(difference), axis=1))
        loss = tf.reduce_mean(tf.nn.relu(euc_dist - params.loss["margin"]) + tf.multiply(tf.cast(euc_dist > params.loss["margin"], dtype=tf.float32), params.loss["margin"]))
    else:
        raise ValueError("Loss strategy type is not recognized: {}".format(type))

    return loss