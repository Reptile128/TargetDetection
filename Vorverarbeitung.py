"""
Die Dateien zum vorverarbeiten der Dateien
"""

import pandas as pd
import re
import string
from langdetect import detect
from deep_translator import GoogleTranslator
import language_tool_python
import spacy
from spacy.lang.lex_attrs import word_shape
from spellchecker import SpellChecker

# Initialisiere die notwendigen Tools
corrector = language_tool_python.LanguageTool('de')  # Rechtschreibkorrektur-Tool für Deutsch
spell = SpellChecker(language="de")
nlp = spacy.load("de_core_news_sm")  # Spacy-Modell für Deutsch

def replace_links(df):
    """
    Ersetzt URLs im Text durch den Platzhalter '[@LINK]'.

    :param df: DataFrame mit der 'text'-Spalte
    :return: Aktualisierter DataFrame
    """
    # Regex-Muster für URLs
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|'
        r'(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    # Ersetze URLs durch '[@LINK]'
    df['text'] = df['text'].apply(
        lambda text: url_pattern.sub("[@LINK]", text) if isinstance(text, str) else text)
    return df


def detect_empty_tweets(df):
    """
    Identifiziert öffentliche Tweets und setzt das 'predicted_label' auf 'public'.

    :param df: DataFrame mit der 'text'-Spalte
    :return: Aktualisierter DataFrame mit der neuen Spalte 'predicted_label'
    """
    invalid_chars = string.digits + string.punctuation + string.whitespace

    def is_public(tweet):
        tweet_cleaned = tweet.strip()
        # Prüft, ob der Tweet nur aus ungültigen Zeichen und '[@LINK]' besteht
        if all(char in invalid_chars for char in tweet_cleaned.replace("[@LINK]", "")) and "[@LINK]" in tweet_cleaned:
            return True
        else:
            return False

    df['predicted_label'] = df['text'].apply(lambda x: 'public' if is_public(x) else None)
    # Entferne '[@LINK]' aus nicht-öffentlichen Tweets
    df.loc[df['predicted_label'].isna(), 'text'] = df.loc[df['predicted_label'].isna(), 'text'].str.replace("[@LINK]",
                                                                                                            "").str.strip()
    return df


def process_hashtags_to_sentence(df):
    """
    Verarbeitet Hashtags, indem sie in Sätze umgewandelt werden.

    :param df: DataFrame mit der 'text'-Spalte
    :return: Aktualisierter DataFrame
    """

    def hashtag_to_sentence(match, is_end_of_sentence):
        hashtag = match.group().lstrip('#')
        # Teilt den Hashtag bei Großbuchstaben
        words = re.findall(r'[A-ZÄÖÜ][a-zäöüß]+', hashtag)
        sentence = " ".join(words)
        if is_end_of_sentence:
            sentence += "."
        return sentence

    def process_tweet(tweet):
        if "#" in tweet:
            processed_tweet = re.sub(
                r'#\w+',
                lambda m: hashtag_to_sentence(m, m.end() == len(tweet)),
                tweet
            )
            return processed_tweet
        else:
            return tweet

    mask = df['predicted_label'] != 'public'
    df.loc[mask, 'text'] = df.loc[mask, 'text'].apply(process_tweet)
    return df


def clean_punctuation(df):
    """
    Bereinigt den Text durch Entfernen unerwünschter Zeichen und Strukturen.

    Folgende Schritte werden durchgeführt:
    - Entfernt Enter, Tab und andere Whitespace-Zeichen, ersetzt durch Leerzeichen.
    - Entfernt Klammern '()' inklusive Inhalt. Bei ungeschlossenen Klammern wird bis zum nächsten Satzzeichen entfernt.
    - Entfernt alle Bindestriche '-'.
    - Entfernt doppelte Satzzeichen.
    - Ersetzt Ausrufezeichen und Fragezeichen durch Punkt.
    - Entfernt alle anderen Satzzeichen außer Komma und Punkt.
    - Entfernt zusätzliche Leerzeichen.

    :param df: DataFrame mit der 'text'-Spalte
    :return: Aktualisierter DataFrame
    """

    def clean_text(text):
        # Entferne Enter, Tab und andere Whitespace-Zeichen, ersetze durch Leerzeichen
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

        # Entferne Klammern und ihren Inhalt
        def remove_parentheses(text):
            # Entfernt Inhalte innerhalb geschlossener Klammern
            text = re.sub(r'\(.*?\)', '', text)
            # Entfernt Inhalte ab ungeschlossener Klammer bis zum nächsten Satzzeichen
            text = re.sub(r'\(.*?[.,]', '', text)
            return text

        text = remove_parentheses(text)

        # Entferne alle Bindestriche '-'
        text = text.replace('-', '')

        # Entferne zusätzliche Leerzeichen
        text = re.sub(r'\s+', ' ', text).strip()

        # Entferne doppelte Satzzeichen
        text = re.sub(r'([^\w\s])\1+', r'\1', text)

        # Ersetze Ausrufezeichen und Fragezeichen durch Punkt
        text = re.sub(r'[!?]', '.', text)

        # Entferne alle anderen Satzzeichen außer ',' und '.'
        text = re.sub(r'[^\w\s,\.]', '', text)

        # Entferne zusätzliche Leerzeichen erneut
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    mask = df['predicted_label'] != 'public'
    df.loc[mask, 'text'] = df.loc[mask, 'text'].apply(clean_text)
    return df


def translate_tweets(df):
    """
    Übersetzt nicht-deutsche Tweets ins Deutsche.

    :param df: DataFrame mit der 'text'-Spalte
    :return: Aktualisierter DataFrame
    """

    def translate_if_not_german(tweet):
        try:
            sprache = detect(tweet)
            if sprache == "de":
                return tweet
            else:
                print(f"Sprache: {sprache}\n{tweet}")
                return GoogleTranslator(source='auto', target='de').translate(tweet)
        except Exception as e:
            print(f"Fehler: {e}")
            return tweet

    mask = df['predicted_label'] != 'public'
    df.loc[mask, 'text'] = df.loc[mask, 'text'].apply(translate_if_not_german)
    return df


def correct_tweets(df):
    """
    Führt eine Rechtschreibkorrektur der Tweets durch.

    :param df: DataFrame mit der 'text'-Spalte
    :return: Aktualisierter DataFrame
    """
    mask = df['predicted_label'] != 'public'

    def correct_and_filter(tweet):
        words = tweet.split()  # Split the sentence into words
        misspelled = spell.unknown(words)  # Find all misspelled words in one go
        corrections = {word: spell.correction(word) for word in misspelled}  # Get corrections for all misspelled words

        # Replace misspelled words with their corrections
        corrected_words = [corrections[word] if word in corrections and corrections[word] is not None else word for word in words]
        return ' '.join(corrected_words)


    df.loc[mask, 'text'] = df.loc[mask, 'text'].apply(correct_and_filter)
    return df


def lemmatize_tweets(df):
    """
    Führt die Lemmatisierung der Tweets durch.

    :param df: DataFrame mit der 'text'-Spalte
    :return: Aktualisierter DataFrame
    """

    def lemmatize_tweet(tweet):
        doc = nlp(tweet)
        lemmatized_sentences = []
        for sent in doc.sents:
            lemmatized_sentence = " ".join(
                [
                    token.lemma_ if not token.is_punct or token.text == "," else token.text
                    for token in sent
                ]
            )
            if not lemmatized_sentence.endswith("."):
                lemmatized_sentence += "."
            lemmatized_sentences.append(lemmatized_sentence)
        return " ".join(lemmatized_sentences)

    mask = df['predicted_label'] != 'public'
    df.loc[mask, 'text'] = df.loc[mask, 'text'].apply(lemmatize_tweet)
    return df


def stop_word_removal(df):
    """
    Entfernt Stoppwörter aus den Tweets.

    :param df: DataFrame mit der 'text'-Spalte
    :return: Aktualisierter DataFrame
    """

    def remove_stop_words(tweet):
        doc = nlp(tweet)
        filtered_tokens = [token.text for token in doc if not token.is_stop or token.is_punct]
        filtered_tweet = " ".join(filtered_tokens)
        return filtered_tweet

    mask = df['predicted_label'] != 'public'
    df.loc[mask, 'text'] = df.loc[mask, 'text'].apply(remove_stop_words)
    return df

def tokenize_and_POS_tweets(df):
    """
    Tokenisiert die Tweets und fügt POS-Tags hinzu.

    :param df: DataFrame mit der 'text'-Spalte
    :return: Aktualisierter DataFrame mit den neuen Spalten 'tokens' und 'pos_tags'
    """

    def tokenize_and_tag(tweet):
        doc = nlp(tweet)
        tokens = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]
        return tokens, pos_tags

    mask = df['predicted_label'] != 'public'
    df.loc[mask, ['tokens', 'pos_tags']] = df.loc[mask, 'text'].apply(
        lambda tweet: pd.Series(tokenize_and_tag(tweet)))
    return df
