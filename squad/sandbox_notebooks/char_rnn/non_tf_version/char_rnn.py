import numpy as np

def random_init(num_rows, num_cols):
    return np.random.rand(num_rows, num_cols) * 0.01

def zero_init(num_rows, num_cols):
    return np.zeros((num_rows, num_cols))

class DataReader:
    def __init__(self, path, seq_length):
        self.fp = open(path, 'r')
        self.data = self.fp.read()
        chars = list(set(self.data))
        self.char_to_ix = {ch: i for (i, ch) in enumerate(chars)}
        self.ix_to_char = {i: ch for (i, ch) in enumerate(chars)}
        self.data_size = len(self.data)
        self.vocab_size = len(chars)
        self.pointer = 0
        self.seq_length = seq_length
    def just_started(self):
        return self.pointer == 0

    def close(self):
        self.fp.close()

    def next_batch(self):
        start = self.pointer
        end = self.pointer + self.seq_length
        inputs = [self.char_to_ix[ch] for ch in self.data[start:end]]
        targets = [self.char_to_ix[ch] for ch in self.data[start+1:end+1]]
        if end + 1 >= self.data_size:
            # reset pointer
            self.pointer = 0
        return inputs, targets


class SimpleRNN:
    def __init__(self, hidden_size, vocab_size, seq_length, learning_rate):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        self.Wxh = random_init(hidden_size, vocab_size) # input to hidden
        self.Whh = random_init(hidden_size, hidden_size) # hidden to hidden
        self.Why = random_init(vocab_size, hidden_size) # hidden to predictions (outputs or y)
        self.bh = zero_init(hidden_size,1)
        self.by = zero_init(vocab_size, 1)

        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

    def forward(self, inputs, hidden_state_prev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hidden_state_prev)
        for t in range(len(inputs)):
            xs[t] = zero_init(self.vocab_size,1)
            char_indx_in_vocabulary = inputs[t]
            xs[t][char_indx_in_vocabulary] = 1 # char_index in vocabulary
            hs[t] = np.tanh(   np.dot(self.Whh, hs[t-1])   +   np.dot(self.Wxh, xs[t])   +   self.bh   ) # hidden state
            ys[t] = np.dot(   self.Why  ,   hs[t]   )  +  self.by # unnormalized log probs for next char
            ps[t] = np.exp(   ys[t]   )  / sum(   np.exp(   ys[t]   )   ) # normalized ys[t]
        return xs, hs, ps

    def backward(self, xs, hs, ps, targets):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        hidden_state_next = zero_init(self.hidden_size, 1)
        for t in reversed(range(len(targets))):
            dy = np.copy(   ps[t]   )
            char_indx_in_vocabulary =targets[t]
            dy[char_indx_in_vocabulary] -= 1 #backprop into y
            dby += dy
            # backproped_y = dy o Transpose of hidden_states starting from very last char which is len(seq.length) - 1
            # so that every char in dy is getting dot producted with every weight in hidden states distributions (such as 100)
            dWhy += np.dot(dy, hs[t].T)

            dh = np.dot(self.Why.T, dy) + hidden_state_next #backprop into h
            dh_raw = ( 1 - hs[t] * hs[t]  ) * dh #backprop through tanh non-linearity
            dbh += dh_raw

            # backpropped hidden (dh) nonlinearity o Transpose of hidden_states starting from previous state
            dWhh += np.dot(dh_raw, hs[t-1].T)

            # backpropped hidden (dh) nonlinearity o Transpose of current char vector starting from very last char
            dWxh += np.dot(dh_raw, xs[t].T)

            hidden_state_next = np.dot(self.Whh.T, dh_raw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            # clip to mitigate exploding gradients between [-5,5]
            np.clip(dparam, -5, 5, out=dparam)

        return dWxh, dWhh, dWhy, dbh, dby

    def loss(self, ps, targets):
        # standard Softmax classifier (also commonly referred to as the cross-entropy loss)
        # for each char in the target list, calculate the -np.log for each chars in each each prediction
        # and sum all of them.
        # such as, let say t = 0, ps[0] means predictions at 0th location
        # we have 21 predictions at 0th location because the vocab size is 21
        # from these 21 predictions, we want to get the prediction of the character we are having
        # in order to get the character index, we need to look at the target's list at position [0]
        # let us say, our char_indx is 15
        # then we are getting the 15th row of the 21 predictions to get the prediction of the char of 15
        # and cont...like that....
        return sum(-np.log(ps[t][targets[t],0]) for t in range(len(targets)))

    def update_model(self, dWxh, dWhh, dWhy, dbh, dby):
        #part of adagrad
        # this loop has a len of 5 since it has 5 members in parallel.
        for param, dparam, mparams in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                          [dWxh, dWhh, dWhy, dbh, dby],
                                          [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]
                                          ):

            # memories for derivatives
            mparams += dparam * dparam

            # update params with learning rate
            param += -self.learning_rate * dparam   /   np.sqrt(mparams + 1e-8)

    def sample(self, h, inputs, character_bucket_size):
        """
        sample a sequence of integers from the model
        inputs is going to be used for seed_ix that is a seed char from the first time step
        h is the memory state,
        n is a new sequence length you want to fill with characters or character_bucket_size
        """
        x = zero_init(self.vocab_size, 1)
        #first char from the input batch let us say 15, and take this 15 and create a new input char by using the one hot vector
        # and make this 15th row as 1 and the rest is 0 so that we can have a sample input
        x[inputs[0]] = 1
        ixes = []
        for t in range(character_bucket_size):
            h = np.tanh(  np.dot(self.Whh, h) + np.dot(self.Wxh, x) + self.bh   )
            y = np.dot( self.Why, h) + self.by
            # having a prediction of this input x
            p = np.exp(y) / np.sum(np.exp(y))
            flattened_p = p.ravel()
            random_char_id_from_predictions =np.random.choice(range(self.vocab_size), p=flattened_p)
            ixes.append(random_char_id_from_predictions)
            # now get the one hot vector of this random selected char and go on with the next iteration
            x = zero_init(self.vocab_size,1)
            x[random_char_id_from_predictions] = 1
        return ixes
    def train(self, data_reader, character_bucket_size):
        iter_num = 0
        smooth_loss = -np.log(1.0/data_reader.vocab_size) * self.seq_length
        while True:
            if data_reader.just_started():
                hidden_prev_state = zero_init(self.hidden_size, 1)
            inputs, targets = data_reader.next_batch()
            xs, hs, ps = self.forward(inputs,hidden_prev_state)
            dWxh, dWhh, dWhy, dbh, dby = self.backward(xs,hs,ps, targets)

            # we are getting the loss of the batch
            loss = self.loss(ps, targets)
            smooth_loss = smooth_loss * 0.999 + loss*0.001

            # now update the model based on the batch for having the new params and memory weights but not derivatives since derivatives are local and has to be recalculated
            self.update_model(dWxh, dWhh, dWhy, dbh, dby)

            # keep getting prev step from the hs values
            hidden_prev_state = hs[self.seq_length - 1]

            if not iter_num%500:
                #in each time how many chars are layed out correctly although they are selected randomly
                sample_ixs = self.sample(hidden_prev_state, inputs, character_bucket_size)
                print('\n\nIter:{}, Loss:{}\n'.format(iter_num, smooth_loss))
                print(''.join(data_reader.ix_to_char[ix] for ix in sample_ixs))
                print(10 * '-')
            iter_num +=1
if __name__ == '__main__':
    seq_length = 25
    data_reader = DataReader("data_min_char_vrnn.txt", seq_length)
    rnn = SimpleRNN(hidden_size=100,
                    vocab_size=data_reader.vocab_size,
                    seq_length=seq_length,
                    learning_rate=1e-1)
    rnn.train(data_reader, 200)