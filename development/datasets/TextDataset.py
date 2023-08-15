from torch.utils.data import Dataset
from torch import tensor

class TextDataset(Dataset):
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