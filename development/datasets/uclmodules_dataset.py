from pandas import read_excel
from numpy import int64
from ast import literal_eval


def load_uclmodules_data(path, only_labled=True):
    data = read_excel(path, sheet_name='Data')
    data = data.drop([
        'Unnamed: 0', 'link', 'heading', 'updated',   'Faculty',
        'Teaching department', 'Credit value', 'Restrictions', 'Timetable',
        'alternative_credit_options', 'Methods of assessment', 'Mark scheme',
        'Number of students on module in previous year', 'Module leader',
        'Who to contact for more information', 'Methods of assessment2',
        'Mark scheme2', 'Number of students on module in previous year2',
        'Module leader2', 'Who to contact for more information2',
        'Teaching location', 'Delivery includes', 'Teaching location2',
        'Delivery includes2', 'Code', 'text_len', 'all_keywords',
        'sdg_keywords', 'SDG_1', 'SDG_2', 'SDG_3', 'SDG_4', 'SDG_5', 'SDG_6', 
        'SDG_7', 'SDG_8', 'SDG_9', 'SDG_10', 'SDG_11','SDG_12', 
        'SDG_13', 'SDG_14', 'SDG_15', 'SDG_16', 'SDG_17', 
    ], axis=1)

    if not only_labled:
        return data
    
    def to_labels(x):
        sdgs_list = []
        
        for sdg in literal_eval(x['final_sdg_labels']):
            label = sdg.split('_')[1]
            sdgs_list.append(int(label))

        return sdgs_list

    data = data[data.astype(str)['final_sdg_labels'] != '[]']
    data['final_sdg_labels'] = data.apply(
        lambda x: to_labels(x),
        axis=1
    )
    data = [
        # data['description'].values,
        data['full_text'].values,
        data['final_sdg_labels'].values
    ]

    return data