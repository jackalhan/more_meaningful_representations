import tensorflow as tf

def euclidean_distance_loss(question_embeddings, paragraph_embeddings, params):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        question_embeddings: question tensor of shape (size, embed_dim)
        paragraph_embeddings: paragraph tensor of shape (size, embed_dim)
        params: parameters that support function
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    euc_dist = euclidean_distance(question_embeddings,paragraph_embeddings)
    if params.loss["version"] == 1:
        subtracted_margin = tf.subtract(euc_dist, params.loss["margin"])
        added_margin = tf.reduce_mean(tf.nn.relu(subtracted_margin)) + params.loss["margin"]
        loss = added_margin
    elif params.loss["version"] == 2:
        loss = tf.reduce_mean(tf.nn.relu(euc_dist - params.loss["margin"]) + tf.multiply(tf.cast(euc_dist > params.loss["margin"], dtype=tf.float32), params.loss["margin"]))
    elif params.loss["version"] == 3:
        loss = tf.reduce_mean(tf.nn.relu(euc_dist - params.loss["margin"]) + tf.multiply(
            tf.cast(euc_dist > params.loss["margin"], dtype=tf.float32), params.loss["margin"]))

    else:
        raise ValueError("Loss strategy type is not recognized: {}".format(type))

    return loss

def euclidean_distance(question_embeddings, paragraph_embeddings):
    """Compute the 2D matrix of distances between all the embeddings.

        Args:
            question_embeddings: question tensor of shape (size, embed_dim)
            paragraph_embeddings: paragraph tensor of shape (size, embed_dim)

        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    difference = tf.subtract(question_embeddings, paragraph_embeddings)
    euc_dist = tf.sqrt(tf.reduce_sum(tf.square(difference), axis=1))
    return euc_dist