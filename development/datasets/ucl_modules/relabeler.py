from development.utils import parse_sdg_id
from pandas import read_excel, ExcelWriter


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