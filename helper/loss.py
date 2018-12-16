import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.utils import pairwise_euclidean_distances, load_embeddings, evaluation_metrics


def euclidean_distance_loss(question_embeddings, paragraph_embeddings, params, labels=None):
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
        tl = Triplet_Loss()
        loss = tl.batch_hard_triplet_loss(question_embeddings, paragraph_embeddings, labels, params.loss["margin"])
    elif params.loss["version"] == 4:
        loss = tf.contrib.losses.metric_learning.npairs_loss(
            labels,
            question_embeddings,
            paragraph_embeddings,
            reg_lambda=params.loss['reg_lambda'],
            print_losses=False
        )

    elif params.loss["version"] == 5:
        loss = tf.contrib.losses.metric_learning.npairs_loss_multilabel(
            labels,
            question_embeddings,
            paragraph_embeddings,
            reg_lambda=params.loss['reg_lambda'],
            print_losses=True
        )
    elif params.loss["version"] == 6:
        tl = Triplet_Loss()
        loss = tl.batch_hard_triplet_loss(question_embeddings, question_embeddings, labels, params.loss["margin"])

        def true_fn_1():
            return 0.0

        def true_fn_2():
            return 30.0

        def false_fn():
            return loss

        loss = tf.cond(loss < 0.0,true_fn_1, false_fn)
        loss = tf.cond(loss > 30.0, true_fn_2, false_fn)
        # p = tf_print(loss,loss , "hello")
        # p.eval()
        #output = tf.Print(loss, [loss], 'Raw Loss:')
    elif params.loss["version"] == 7:
        tl = Triplet_Loss()
        loss_q_tp_p = tl.batch_hard_triplet_loss(question_embeddings, paragraph_embeddings, labels, params.loss["margin"])
        loss_q_to_q = tl.batch_hard_triplet_loss(question_embeddings, question_embeddings, labels, params.loss["margin"])
        loss = loss_q_tp_p + loss_q_to_q
    else:
        raise ValueError("Loss strategy type is not recognized: {}".format(type))

    return loss

# def tf_print(op, tensors, message=None):
#     def print_message(x):
#         sys.stdout.write(message + " %s\n" % x)
#         return x
#
#     prints = [tf.py_func(print_message, [tensor], tensor.dtype) for tensor in tensors]
#     with tf.control_dependencies(prints):
#         op = tf.identity(op)
#     return op

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

class  Triplet_Loss(object):
    """Define functions to create the triplet loss with online triplet mining."""
    def _get_anchor_positive_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]

        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

        # Combine the two masks
        mask = tf.logical_and(indices_not_equal, labels_equal)

        return mask

    def _get_anchor_pure_positive_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]

        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

        # Combine the two masks
        mask = tf.logical_and(indices_equal, labels_equal)

        return mask

    def _get_anchor_negative_triplet_mask(self, labels):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]

        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

        mask = tf.logical_not(labels_equal)

        return mask

    def _get_triplet_mask(self, labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check that i, j and k are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

        distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        i_equal_j = tf.expand_dims(label_equal, 2)
        i_equal_k = tf.expand_dims(label_equal, 1)

        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

        # Combine the two masks
        mask = tf.logical_and(distinct_indices, valid_labels)

        return mask

    def batch_all_triplet_loss(self, question_embeddings, paragraph_embeddings, labels, margin):
        """Build the triplet loss over a batch of embeddings.

        We generate all the valid triplets and average the loss over the positive ones.

        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = pairwise_euclidean_distances(question_embeddings, paragraph_embeddings)
        #labels = tf.reduce_mean(paragraph_embeddings, axis=1)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = self._get_triplet_mask(labels)
        mask = tf.to_float(mask)
        triplet_loss = tf.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets

    def batch_hard_triplet_loss(self,question_embeddings, paragraph_embeddings, labels, margin ):
        """Build the triplet loss over a batch of question and paragraph embeddings.

        For each anchor, we get the hardest positive and hardest negative to form a triplet.

        Args:
            ....

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix between question and paragraphs
        pairwise_dist  = pairwise_euclidean_distances(question_embeddings, paragraph_embeddings)

        # For each question, get the hardest positive paragraphs
        # First, we need to get a mask for every valid positive paragraphs (must have same label)
        # The following one provides a 2D mask where mask[question, positive paragraph] is True iff
        # question and positive paragraph are distinct and
        # have same label
        mask_anchor_positive = self._get_anchor_pure_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)

        # Make 0 any element where (question, positive paragraph) is not valid
        # (valid if question != positive paragraph, label(question) == label(positive paragraph))
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

        # shape (batch_size, 1)
        positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(positive_dist))

        # For each anchor, get the hardest negative paragraphs
        # First, we need to get a mask for every valid negative paragraph (must have different labels)
        # The following one provides a 2D mask where mask[question, negative paragraph] is True iff
        # question and negative question have distinct labels
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)

        # Add the maximum value in each row to the invalid negative paragraph
        # (label(question) == label(negative paragraph))
        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
        tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

        if margin is not None:
            # Combine largest d(question, positive paragraphs) and lowest d(question, negative paragraphs) into final triplet loss
            triplet_loss = tf.nn.relu(positive_dist - hardest_negative_dist + margin)
        else:
            pass

        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss