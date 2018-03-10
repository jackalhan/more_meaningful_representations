import os
from bilm.elmo import ElmoEmbedder
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.blank("en")

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

_basepath = '/home/jackalhan/Development/github/more_meaningful_representations/squad/'
dataset_type = 'dev'
dataset_version = 'v1.1'

_input_file_name = 'input_words.txt'
_neighbor_words_file_name =  'nearest_words.csv'

datadir = os.path.join(_basepath)
neighbor_words = os.path.join(datadir, _neighbor_words_file_name)
input_file = os.path.join(datadir, _input_file_name)


with open(input_file) as f:
    contents = f.readlines()
#_contents = [[content.strip()] for content in contents]
contents = [word_tokenize(content.strip()) for content in contents]



ee = ElmoEmbedder()

slices = [{'slice_type':'All', 'slice_index':None, 'axis':(1,2)},
          {'slice_type':'1st', 'slice_index':0, 'axis':(1)},
          {'slice_type':'2nd', 'slice_index':1, 'axis':(1)},
          {'slice_type':'3rd', 'slice_index':2, 'axis':(1)}]
neighbor_list = []
for _s in slices:
    print('Processing : {}'.format(_s))
    # embeddings_ = embeddings
    # if _s['slice_index'] is not None:
    #     embeddings_ = embeddings[:,_s['slice_index']]
    # embeddings_ = np.apply_over_axes(np.mean, embeddings_, _s['axis'])
    #embeddings_ = np.reshape(embeddings_,(embeddings.shape[0], embeddings.shape[3]))
    embeddings = np.asarray(ee.list_to_embeddings(contents, _s['slice_index']))
    embeddings = np.reshape(embeddings, (embeddings.shape[0], ee.dims))
    for _id, _embedding in enumerate(tqdm(embeddings, total=len(embeddings))):
        _embedding = np.array([_embedding])
        sk_sim = cosine_similarity(_embedding,embeddings)[0]
        neighbors = np.argsort(-sk_sim)
        for _, neighbor_id in enumerate(neighbors[0:5]):
            neighbor_list.append((_s['slice_type'], " ".join(contents[_id]), " ".join(contents[neighbor_id]),_+1, sk_sim[neighbor_id]))
df_neighbors = pd.DataFrame(data=neighbor_list, columns=['slice_type','word', 'neighbor_word', 'neighbor_order', 'cos_similarity'])
df_neighbors.to_csv(neighbor_words, index=False)
print('Completed')

