import re


def tokenize_sentence(text: str) -> list:
    """
    :param text: text which is needed to be tokenized
    :return: list of tokens
    """
    return re.findall(r'[\w]+', text.lower())


def tokenize_corpus(corpus: list) -> list:
    """
    :param corpus: corpus of sentences
    :return: list of tokenized sentences
    """
    return [tokenize_sentence(text) for text in corpus]


def join_tokenized_sentences(tokenized_sentences: list) -> list:
    """
    :param tokenized_sentences: tokenized corpus
    :return: corpus of sentences
    """
    sentence_list = []
    for doc in tokenized_sentences:
        sentence = str()
        for token in doc:
            sentence += token + ' '
        sentence_list.append(sentence.rstrip())
    return sentence_list
