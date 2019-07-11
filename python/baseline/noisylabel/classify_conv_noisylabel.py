from baseline.pytorch.classify import ConvModel, append2seq
import torch.nn as nn
from collections import OrderedDict



class ConvNoiseModel(ConvModel):

    """Class extends the base model but adds a final linear layer and softmax to account for noisy labels

    """
    def __init__(self):
        super(ConvNoiseModel, self).__init__()

    def _init_output(self, input_dim, nc):
        self.output = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(input_dim, nc)),
            ('softmax', nn.Softmax(dim=1)),
            ('linear2', nn.Linear(nc, nc, bias=False)),
            # ('drop1', nn.Dropout(p=0.2)),
            ('logSoftmax', nn.LogSoftmax(dim=1))]))


def create_model(embeddings, labels, **kwargs):
    return ConvNoiseModel.create(embeddings, labels, **kwargs)


def load_model(modelname, **kwargs):
    return ConvNoiseModel.load(modelname, **kwargs)

