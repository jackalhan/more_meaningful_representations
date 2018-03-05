### Issue 1 : OSError: Can't find model 'en'

#### Solution : 

python -m spacy download en_core_web_sm

python -m spacy download en

### Warnings: 

- You can download pretrained highway weights from the following link into the model folder of the project in your local disk:
  
  * [weight file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)
  

- Unnecessary dependency for allennlp, it has nothing to do with our repo but we need to install it not to get an error.

  pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp35-cp35m-linux_x86_64.whl