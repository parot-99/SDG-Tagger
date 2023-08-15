from pandas import read_csv
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split


def load_relx_data(data_path, training=False):
    data = read_csv(data_path)
    train_ratio = 0.75
    test_ratio = 0.15
    valid_ratio = 0.15


    texts = data['full_abstract'].values
    sdgs = data[[f'sdg_{i}' for i in range(1, 17)]].values

    if not training:
        return [texts, sdgs]

    train_data, test_data, train_labels, test_labels = train_test_split(
        texts,
        sdgs,
        test_size=1 - train_ratio,
        random_state=47,
        shuffle=True
    )

    valid_data, test_data, valid_labels, test_labels = train_test_split(
        test_data,
        test_labels,
        test_size=test_ratio/(test_ratio + valid_ratio),
        random_state=47
    )

    data = {
        'train': [train_data, train_labels],
        'valid': [valid_data, valid_labels],
        'test': [test_data, test_labels]
    }

    return data
