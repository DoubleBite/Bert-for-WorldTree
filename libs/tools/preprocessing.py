import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


class Lemmatizer:
    pass


def get_keyword_tokens(input_string):

    # Replace punctuations
    string = re.sub(r'[^\w\s]', ' ', input_string)

    # Normalize words
    words = string.split()
    words = [w.lower() for w in words]

    # Remove stop words
    filtered_words = [w for w in words if not w in stop_words]

    return set(filtered_words)
