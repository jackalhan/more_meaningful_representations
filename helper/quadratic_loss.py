import tensorflow as tf

def euclidean_distance_loss(embeddings, target_embeddings, params, type='square'):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    difference = tf.subtract(embeddings, target_embeddings)
    if type == 'square':
        loss = tf.reduce_mean(tf.square(difference))
    else:
        loss = tf.subtract(tf.sqrt(tf.reduce_sum(tf.square(difference), axis=1)), params.margin)
        loss = tf.reduce_mean(tf.nn.relu(loss)) + params.margin
        return loss