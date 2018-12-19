### Warnings: 

- You can download pretrained highway weights from the following link into the model folder of the project in your local disk:
  
  * [weight file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)
  

### Issue 1 : OSError: Can't find model 'en'

#### Solution : 

(1)python -m spacy download en_core_web_sm

(2) python -m spacy download en

### if you have any problem during the sudo apt-get update command

#### Solution :
We need to comment google sdk line by typing the following command
sudo gedit /etc/apt/sources.list.d/google-cloud-sdk.list

and then 
install tf-ranking by following the guideline from the given link
https://github.com/tensorflow/ranking

pip install --upgrade tensorflow
pip install --upgrade tensorflow-gpu==1.9.0

### if you have some strange problem for gensim:
#### Solution :
pip install -U boto
