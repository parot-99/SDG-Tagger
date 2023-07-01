from pandas import read_csv
from json import loads
from numpy import empty

def load_videscription_data(path):
    vid_data = read_csv(path)
    length = vid_data.shape[0]
    data = empty(length, dtype=object)

    for i in range(length):
        data[i] = loads(vid_data['data'][i])['description']

    data = list(filter(None, data))

    return data
