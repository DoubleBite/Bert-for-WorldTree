import re


def parse_raw_question(raw_question: str):
    """
    Becuase originally the question text and choices are concatenated together, we need to separate them.

        Example Usage
        ---------------------
        >>> raw_question = ('Students visited the ... Which statement best explains why the sun appears to move '
        ... 'across the sky each day? (A) The sun revolves around Earth. (B) Earth rotates around the sun. '
        ... '(C) The sun revolves on its axis. (D) Earth rotates on its axis')
        >>> question, choices = parse_raw_question(raw_question)
        >>> question
        'Students visited the ... Which statement best explains why the sun appears to move across the sky each day?'
        >>> choices # doctest: +NORMALIZE_WHITESPACE
        ['The sun revolves around Earth.', 'Earth rotates around the sun.', 
        'The sun revolves on its axis.', 'Earth rotates on its axis']
    """
    trunks = re.split(r"\([ABCDE12345]\)", raw_question)
    trunks = [x.strip() for x in trunks]
    question, *choices = trunks

    return question, choices


def answser_to_index(answer: str):
    """
    Convert the answer to its index. 
    For example, when the answer is A, it means the 0th choice. Hence, we return 0.
    """
    mapping = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4,
    }

    return mapping[answer]
