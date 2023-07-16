import spacy

class RobertaNER:
    def __init__(self) -> None:
        self.__model = spacy.load('en_core_web_trf')

    def predict(self, text):
        return self.__model(text)
    
    def print_entities(self, text, entities=['ORG']):
        doc = self.__model(text)

        for entity in doc.ents:
            if entity.label_ in entities:
                print(f'{entity.label_}: {entity.text}')