# Keyphrase Extraction using BERT

Deep Keyphrase extraction using BERT.

## Run in Colab

<a href="https://colab.research.google.com/drive/1MIZHsnsscPK96Sh6va1-LP6ODxidj4Er?usp=sharing">Link to the Notebook</a>

## Usage

1. Clone this repository and install `pytorch-pretrained-BERT`
2. Change the parameters accordingly in `experiments/base_model/params.json`. We recommend keeping batch size of 4 and sequence length of 512, with 6 epochs, if GPU's VRAM is around 11 GB.
3. For training, run the command `python train.py`
4. For eval, run the command, `python evaluate.py`
5. For running prediction on `data/h1_7.txt` file, run `python keyphrase_task.py`

Python version 3.7

## Results

### Subtask 1: Keyphrase Boundary Identification Using BERT

We used IO format here. 

On test set, we got:

1. **F1 score**: 0.3799
2. **Precision**: 0.2992
3. **Recall**: 0.5201
4. **Support**: 921

### Prediction on the given text file in `data/h1_7.txt`

['I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

## Changes after the test on Aug 7, 2021

- Implemented a custom BERT Model, applying Bi-LSTM and CRF layers over PyTorch's `bert-base-uncased` pre-trained model
- Added a weight matrix to the loss function to implement `weighted CrossEntropyLoss`

## Credits

This is a modified version of the original repo `pranav-ust/BERT-keyphrase-extraction`

