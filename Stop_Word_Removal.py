import spacy

nlp = spacy.load("de_core_news_sm")  # Spacy-Modell für Deutsch
def stop_word_removal(df):
    """
    Entfernt Stoppwörter aus den Tweets.

    :param df: DataFrame mit der 'text'-Spalte
    :return: Aktualisierter DataFrame
    """

    def remove_stop_words(tweet):
        doc = nlp(tweet)
        # Entferne nur, wenn das Wort ein Stoppwort ist, aber kein Pronomen
        filtered_tokens = [
            token.text for token in doc if not (token.is_stop and token.pos_ != "PRON")
        ]
        filtered_tweet = " ".join(filtered_tokens)
        return filtered_tweet

    mask = df['predicted_label'] != 'public'
    df.loc[mask, 'text'] = df.loc[mask, 'text'].apply(remove_stop_words)
    return df