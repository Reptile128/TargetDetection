import time
from Load_Safe import load_data, save_processed_data, load_processed_data
from Vorverarbeitung import (
    replace_links,
    detect_empty_tweets,
    process_hashtags_to_sentence,
    translate_tweets,
    correct_tweets,
    lemmatize_tweets,
    tokenize_and_POS_tweets,
    clean_punctuation)
from Stop_Word_Removal import stop_word_removal

Pfad_zur_Originaldatei = "Data/Tweets_Original.csv"
Pfad_zur_verarbeiteten_Datei = "Data/Tweets_Complete_Preprocessing.csv"


def measure_time(func, *args, **kwargs):
    """
    Führt eine Funktion aus und misst die benötigte Zeit.
    """
    print(f"Starte {func.__name__}")
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    elapsed_time = end - start
    print(f"{func.__name__} dauerte {elapsed_time:.2f} Sekunden.")
    return result


def main():
    # Startzeitpunkt
    start_time = time.time()

    # Laden der Daten
    df = measure_time(load_data, Pfad_zur_Originaldatei)

    # Links ersetzen
    df = measure_time(replace_links, df)

    # Vorverarbeitungsschritte
    df = measure_time(detect_empty_tweets, df)
    df = measure_time(process_hashtags_to_sentence, df)
    df = measure_time(translate_tweets, df)
    df = measure_time(correct_tweets, df)
    df = measure_time(clean_punctuation, df)
    df = measure_time(lemmatize_tweets, df)
    df = measure_time(stop_word_removal, df)
    df = measure_time(tokenize_and_POS_tweets, df)

    # Konvertiere die Listen in Strings für das Speichern
    df['tokens'] = df['tokens'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
    df['pos_tags'] = df['pos_tags'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')

    # Endzeitpunkt
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Die gesamte Vorverarbeitung dauerte {total_time:.2f} Sekunden.")

    # Speichern der verarbeiteten Daten mit Anführungszeichen und allen Spalten
    save_processed_data(df, Pfad_zur_verarbeiteten_Datei)


if __name__ == "__main__":
    main()

