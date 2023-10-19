from torch.utils.data import Dataset
from torch import tensor

class TextDataset(Dataset):
    """ A custom class that inherits Pytorch's Dataset class. Used for loading the data to torch models for training and inference.

    Attributes:
    -----------
    encodings: str
        enocdings of the text data, aquired by running texts through an LLM tokenizer
    labels: int
        SDG label-encoded representation (Note: The 16 SDGs are labled as 0-15 due to transformers only accepting an input that starts from 0)

    Methods:
    --------
    __len__:
        returns the size of data
    __getitem__:
        returns a sample from the data using an index
    """
    
    def __init__(self, encodings, labels):
        self.__encodings = encodings
        self.__labels = labels

    def __len__(self):
        return len(self.__labels)

    def __getitem__(self, idx):
        sample = {
            key: val[idx].clone().detach() for key, val in self.__encodings.items()
        }
        sample['labels'] = tensor(self.__labels[idx])

        return sample