import spacy

class RobertaNER:
    def __init__(self) -> None:
        self.__model = spacy.load('en_core_web_trf')

    def predict(self, text):
        return self.__model(text)
    
    def print_entities(self, text, include_entities=['ORG']):
        doc = self.__model(text)

        for entity in doc.ents:
            if entity.label_ in include_entities:
                print(f'{entity.label_}: {entity.text}')

    def get_entities(self, text, include_entities=['ORG']):
        doc = self.__model(text)
        entities = ''

        for entity in doc.ents:
            if entity.label_ in include_entities:
                entities += f'{entity.label_}: {entity.text} - '

        return entities