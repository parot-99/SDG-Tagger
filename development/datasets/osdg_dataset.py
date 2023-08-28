from pandas import read_csv
from sklearn.model_selection import train_test_split


def load_osdg_data(data_path, training=False, filter_agreement=False):
    data = read_csv(data_path, delimiter=r'\t', engine='python')
    train_ratio = 0.70
    test_ratio = 0.15
    valid_ratio = 0.15

    if filter_agreement:
        data = data[data['agreement'] >= 0.6]

    if not training:
        return data

    texts = data['text'].values
    sdgs = data['sdg'].values - 1

    train_data, test_data, train_labels, test_labels = train_test_split(
        texts,
        sdgs,
        test_size=1 - train_ratio,
        stratify=sdgs,
        random_state=47,
        # random_state=47,
        shuffle=True
    )

    valid_data, test_data, valid_labels, test_labels = train_test_split(
        test_data,
        test_labels,
        test_size=test_ratio/(test_ratio + valid_ratio),
        stratify=test_labels,
        random_state=47
    )

    data = {
        'train': [train_data, train_labels],
        'valid': [valid_data, valid_labels],
        'test': [test_data, test_labels]
    }

    return data
