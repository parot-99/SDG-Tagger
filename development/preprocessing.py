def remove_stop_words(texts, mode='list'):
    stop_words = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    }

    if mode != 'list':
        text = texts.lower().split(' ')
        text_no_stopwords_list = [
            word for word in text if word not in stop_words
        ]
        text_no_stopwords = ' '.join(text_no_stopwords_list)

        return text_no_stopwords
    
    for i, text in enumerate(texts):
        text = text.lower().split(' ')
        text_no_stopwords_list = [
            word for word in text if word not in stop_words
        ]
        text_no_stopwords = ' '.join(text_no_stopwords_list)
        texts[i] = text_no_stopwords

    return texts