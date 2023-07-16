import pandas as pd
from .utils import parse_label

def full_pipe(tagger_model, ner_model, sentiment_model=None, texts=[]):
    full_pipe_frame = pd.DataFrame(columns=[
        'Text',
        'SDG',
        'Entities',
        'Sentiment'
    ])

    for i, text in enumerate(texts):
        sdg = tagger_model.predict(text)
        sdg_label = parse_label(sdg)
        entities = ner_model.get_entities(text)

        full_pipe_frame.loc[i] = [
            text,
            sdg_label,
            entities,
            'NULL'
        ]

    full_pipe_frame.style.set_properties(**{'text-align': 'left'})

    return full_pipe_frame
