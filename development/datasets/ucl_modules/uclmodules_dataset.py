from pandas import read_excel, ExcelWriter
from numpy import where
from ast import literal_eval
from development.utils import parse_sdg_id


def load_uclmodules_data(path, only_labled=True, evaluation=False):
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
        'sdg_keywords', 
    ], axis=1)

    if only_labled:
        data = data[data.astype(str)['final_sdg_labels'] != '[]']

    if evaluation:
        sdgs = data[[f'SDG_{i}' for i in range(1, 17)]].values
        sdgs = where(sdgs == 'Yes', 1, 0)
        data = [
            # data['description'].values,
            data['full_text'].values,
            sdgs
        ]

        return data
    
    def to_labels(x):
        sdgs_list = []
        
        for sdg in literal_eval(x['final_sdg_labels']):
            label = sdg.split('_')[1]
            sdgs_list.append(int(label) - 1)

        return sdgs_list

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


def relabel_ucl_data(data_path, model, output_path):
    data = read_excel(data_path, sheet_name='Data')
    data = data.drop(['all_keywords', 'sdg_keywords'], axis=1)
    data = data.drop(['Unnamed: 0'], axis=1)
    data['final_sdg_labels'] = ''
    data[[f'SDG_{i}' for i in range(1, 17)]] = 0
    module_texts = data['full_text'].values

    predictions = model.cls_pipeline(
        module_texts,
        parse=True, 
        top_k=16,
        threshold=0.7
    )

    for index, row in data.iterrows():
        for prediction in predictions[index]:
            fina_sdg_label = parse_sdg_id(prediction - 1) + ' -'
            data.at[index, 'final_sdg_labels'] += fina_sdg_label
            data.at[index, f'SDG_{prediction}'] = 1

    with ExcelWriter(output_path, mode='w') as writer:
        data.to_excel(writer, sheet_name='Data', index=False)