import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert import BertModel


class CustomBERTModel(BertModel):
    """
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].
    """

    def __init__(self, config, num_labels):
        super(CustomBERTModel, self).__init__(config)
        self.num_labels = num_labels
        # self.bert = BertModel(config)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config_hidden_size = config.hidden_size

        # Additional layers
        self.hidden_dim = 256
        # Bi-LSTM layer
        self.rnn = nn.LSTM(self.config_hidden_size,
                           self.hidden_dim, bidirectional=True)
        # Linear layer
        self.classifier = nn.Linear(self.hidden_dim * 2, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        # sequence_output -> [batch_size, sequence_length, 768]

        # Additional layers
        # Bi-LSTM
        lstm_output, (hidden, cell) = self.rnn(sequence_output)
        # lstm_output = [batch_size, sequence_length, 256 * 2]

        # hidden = torch.cat(
        #     (lstm_output[:, :, :256], lstm_output[:, :, 256:]), dim=-1)
        # logits = self.classifier(hidden.view(-1, self.hidden_dim * 2))
        logits = self.classifier(lstm_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
