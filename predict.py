"""Use the model to predict"""

import os
import random
import numpy as np
import torch
import utils

from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertForTokenClassification
from model import CustomBERTModel

# Load the parameters from json file
json_path = 'experiments/base_model/params.json'
assert os.path.isfile(
    json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)
params.device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')

# Custom data-loader function


class DataLoader(object):
    def __init__(self, data_dir, bert_model_dir, params, token_pad_idx=0):
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.token_pad_idx = 0

        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag

        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model_dir, do_lower_case=True)

    def load_tags(self):
        tags = []
        file_path = 'data/task1/tags.txt'
        with open(file_path, 'r') as file:
            for tag in file:
                tags.append(tag.strip())
        return tags

    def load_sentences_tags(self, sentences_file, d):
        """Loads sentences from their corresponding files. 
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        sentences = []

        with open(sentences_file, 'r') as file:
            for line in file:
                # replace each token by its index
                tokens = line.split()
                sentences.append(self.tokenizer.convert_tokens_to_ids(tokens))

        # storing sentences and tags in dict d
        d['data'] = sentences
        d['size'] = len(sentences)

    def load_data(self):
        """Loads the data from data_dir.

        Returns:
            data: (dict) contains the data
        """
        data = {}

        sentences_file = 'data/h1_7.txt'
        self.load_sentences_tags(sentences_file, data)

        return data

    def data_iterator(self, data, shuffle=False):
        """Returns a generator that yields batches data with tags.

        Args:
            data: (dict) contains data which has keys 'data', 'size'
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (tensor) shape: (batch_size, max_len)
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(self.seed)
            random.shuffle(order)

        # one pass over data
        for i in range(data['size']//self.batch_size):
            # fetch sentences and tags
            sentences = [data['data'][idx]
                         for idx in order[i*self.batch_size:(i+1)*self.batch_size]]

            # batch length
            batch_len = len(sentences)

            # compute length of longest sentence in batch
            batch_max_len = max([len(s) for s in sentences])
            max_len = min(batch_max_len, self.max_len)

            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_data = self.token_pad_idx * np.ones((batch_len, max_len))

            # copy the data to the numpy array
            for j in range(batch_len):
                cur_len = len(sentences[j])
                if cur_len <= max_len:
                    batch_data[j][:cur_len] = sentences[j]
                else:
                    batch_data[j] = sentences[j][:max_len]

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)

            # shift tensors to GPU if available
            batch_data = batch_data.to(self.device)

            yield batch_data


# Instantiate the DataLoader
data_loader = DataLoader(
    'data/h1_7.txt', 'bert-base-uncased', params, token_pad_idx=0)

# Load data
pred_data = data_loader.load_data()
pred_data_iterator = next(data_loader.data_iterator(pred_data))
data_masks = pred_data_iterator.gt(0)


# Define the model
config_path = 'model/bert_config.json'
config = BertConfig.from_json_file(config_path)
model = CustomBERTModel(config, num_labels=len(params.tag2idx))
# model = BertForTokenClassification(config, num_labels=len(params.tag2idx))

model.to(params.device)
# Reload weights from the saved file
utils.load_checkpoint(
    'experiments/base_model/best.pth.tar', model)

output = model(pred_data_iterator, token_type_ids=None,
               attention_mask=data_masks)
output = output.detach().cpu().numpy()

pred_tags = []
pred_tags.extend([params.idx2tag.get(idx) for indices in np.argmax(
    output, axis=2) for idx in indices])
print(pred_tags)
