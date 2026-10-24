"""Prompt sanitization identical to the reference ``kimodo.sanitize``.

Prompt embeddings are stored under their sanitized text, so lookups and
the encoder must agree on this normalization.
"""

FINAL_PUNCTUATION = ".!?\"])'"


def sanitize_text(text):
    text = " ".join(text.split())
    if text == "":
        return text
    text = strip_leading_symbols(text).capitalize()
    text = strip_trailing_symbols(text)
    if text[-1] not in ".!?":
        text = text + "."
    return capitalize_sentences(text)


def strip_leading_symbols(text):
    for index, character in enumerate(text):
        if character.isalnum():
            return text[index:]
    return text


def strip_trailing_symbols(text):
    for index in range(len(text) - 1, -1, -1):
        character = text[index]
        if character.isalnum() or character in FINAL_PUNCTUATION:
            return text[:index + 1]
    return text


def capitalize_sentences(text):
    for sentence_break in ".!?":
        pieces = [piece.strip() for piece in text.split(sentence_break)]
        pieces = [piece[:1].upper() + piece[1:] for piece in pieces]
        text = f"{sentence_break} ".join(pieces).strip()
    return text
