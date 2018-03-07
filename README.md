### Warnings: 

- You can download pretrained highway weights from the following link into the model folder of the project in your local disk:
  
  * [weight file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)
  

### Issue 1 : OSError: Can't find model 'en'

#### Solution : 

(1)python -m spacy download en_core_web_sm

(2) python -m spacy download en

