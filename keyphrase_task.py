"""Use the model to predict"""

import os
import numpy as np
import torch
from pytorch_pretrained_bert import BertForTokenClassification, BertConfig
from data_loader import DataLoader
import utils

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

n_gram_range = (1, 1)
stop_words = "english"

# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range,
                        stop_words=stop_words).fit('data/h1_7.txt')
candidates = count.get_feature_names()


# Load the parameters from json file
json_path = 'experiments/base_model/params.json'
assert os.path.isfile(
    json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)
# Define the model
# config_path = 'bert-base-uncased/bert_config.json'
config_path = 'model/bert_config.json'
config = BertConfig.from_json_file(config_path)
model = BertForTokenClassification(config, num_labels=len(params.tag2idx))
model.to(params.device)
# Reload weights from the saved file
utils.load_checkpoint(
    'experiments/base_model/best.pth.tar', model)
if args.fp16:
    model.half()
if params.n_gpu > 1 and args.multi_gpu:
    model = torch.nn.DataParallel(model)

doc_embedding = model.encode('data/h1_7.txt')
candidate_embeddings = model.encode(candidates)

# Cosine similarity
top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

print(keywords)
