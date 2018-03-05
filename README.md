Issue : OSError: Can't find model 'en'
Solution : python -m spacy download en_core_web_sm
           python -m spacy download en

- Unnecessary dependency for allennlp, it has nothing to do with our repo but we need to install it not to get an error.
pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp35-cp35m-linux_x86_64.whl