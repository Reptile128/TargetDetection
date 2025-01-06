import pandas as pd
import re
from langdetect import detect, DetectorFactory, LangDetectException
from deep_translator import GoogleTranslator
import spacy
import csv
from spellchecker import SpellChecker
from tqdm import tqdm

# Setzt den Startwert von LangDetect damit die ergebnisse konstant sind
DetectorFactory.seed = 0
# Initiere den Spellchecker
spell = SpellChecker(language="de")
# Lade das deutsche SpaCy-Modell
try:
    nlp = spacy.load('de_core_news_sm')
except OSError:
    print("Das SpaCy-Modell 'de_core_news_sm' ist nicht installiert. Bitte installiere es mit:")
    print("python -m spacy download de_core_news_sm")
    exit(1)

def preprocess_text(text):
    """
    Fuehrt die folgenden Schritte zur Textvorverarbeitung durch:
    1. Entfernt Links.
    2. Entfernt Hashtags.
    3. Ersetzt Satzzeichen gemäß den Vorgaben.
    4. Entfernt Inhalte innerhalb von Klammern und Anführungszeichen.
    5. Entfernt Enter, Tabs, Zahlen und redundante Satzzeichen.

    Args:
        text(String): Die einzelnen Eintraege der CVS Datei

    Returns:
        text(String): Die fertig verarbeiteten Eintraege
    """

    if pd.isnull(text):
        return ""

    # 1. Entferne Links
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # 2. Entferne Hashtags
    text = re.sub(r'#\w+', '', text)

    text = detect_and_translate(text)

    # 3. Ersetze ?! und ähnliche durch Punkt
    text = re.sub(r'[?!]+', '.', text)

    # 4. Entferne alle anderen Satzzeichen außer Punkt
    text = re.sub(r'[^\w\s\.,\[\]@]', '', text)

    # 5. Entferne Inhalte innerhalb von "" oder ()
    def remove_within(text, open_char, close_char):
        open_esc = re.escape(open_char)
        close_esc = re.escape(close_char)
        pattern = re.compile(r'{open_esc}[^\{close_char}\.]*?(\{close_char}|\.)')
        while True:
            match = pattern.search(text)
            if not match:
                break
            end = match.end()
            replacement = '.' if match.group(1) == '.' else ''
            text = text[:match.start()] + replacement + text[end:]
        return text

    text = remove_within(text, '"', '"')
    text = remove_within(text, '(', ')')

    # 6. Entferne Enter und Tabs
    text = text.replace('\n', ' ').replace('\t', ' ')

    # 7. Entferne Zahlen
    text = re.sub(r'\d+', '', text)

    # 8. Ersetze Muster wie '. .' oder '... ' durch einen einzigen Punkt
    text = re.sub(r'\.\s*\.', '.', text)
    text = re.sub(r'\.{2,}', '.', text)

    # 9. Entferne überflüssige Leerzeichen
    text = re.sub(r'\s+', ' ', text).strip()

    # 10. Nur Kleinschreibung
    text = text.lower()

    return text

def detect_and_translate(text):
    """
    Erkennt die Sprache des Textes. Wenn er nicht Deutsch ist, uebersetzt es ins Deutsche. Falls der Text bereits auf deutsch sein sollte, wird die Rechtschreibung der Woerter kontrolliert.

    Args:
        text(String): Die einzelnen Eintraege der CVS Datei

    Returns:
        text(String): Die uebersetzten oder berichtigten Eintraege
    """
    if not text:
        return text
    try:
        lang = detect(text)
    except LangDetectException:
        lang = 'de'

    if lang != 'de':
        try:
            translated = GoogleTranslator(source='auto', target='de').translate(text)
            return translated
        except Exception as e:
            print(f"Übersetzungsfehler: {e}")
            return text
    else:
        words = text.split()
        misspelled = spell.unknown(words)
        corrections = {word: spell.correction(word) for word in misspelled}
        corrected_words = [corrections[word] if word in corrections and corrections[word] else word for word in words]
        return ' '.join(corrected_words)

def remove_adjectives(text):
    """
    Entfernt alle Adjektive aus dem Text.

    Args:
        text(String): Die einzelnen Eintraege der CVS Datei

    Returns:
        text(String): Die Eintraege ohne Adjektive
    """
    if not text:
        return text
    doc = nlp(text)
    tokens = [token.text for token in doc if token.pos_ != 'ADJ']
    return ' '.join(tokens)

def main():
    print("Beginne mit der Vorverarbeitung der Beschreibungen...")

    try:
        df = pd.read_csv('Data/Tweets_Original.csv', sep=';', quotechar='"', encoding='utf-8', quoting=csv.QUOTE_ALL)
    except FileNotFoundError:
        print("Die Datei wurde nicht gefunden.")
        return
    except Exception as e:
        print(f"Fehler beim Einlesen der Datei: {e}")
        return

    if not {'id', 'description', 'TAR'}.issubset(df.columns):
        print("Die Eingabedatei muss die Spalten 'id', 'description' und 'TAR' enthalten.")
        return

    print("Starte die Textvorverarbeitung...")
    tqdm.pandas(desc="Verarbeite Beschreibungen")
    df['clean_description'] = df['description'].progress_apply(preprocess_text)

    print("Speichere vorverarbeitete Datei...")
    df_verarbeitet = df[['id', 'clean_description', 'TAR']].rename(columns={'clean_description': 'translated_description'})
    try:
        df_verarbeitet.to_csv('Data/D_verarbeitet.csv', sep=';', quotechar='"', quoting=csv.QUOTE_ALL, index=False, encoding='utf-8')
        print("D_verarbeitet.csv wurde erfolgreich gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern von D_verarbeitet.csv: {e}")

    print("Erstelle eine Version ohne Adjektive...")
    df_verarbeitet['description_oa'] = df_verarbeitet['translated_description'].progress_apply(remove_adjectives)
    df_verarbeitet_oa = df_verarbeitet[['id', 'description_oa', 'TAR']].rename(columns={'description_oa': 'description'})
    try:
        df_verarbeitet_oa.to_csv('Data/D_verarbeitet_oa.csv', sep=';', quotechar='"', quoting=csv.QUOTE_ALL, index=False, encoding='utf-8')
        print("D_verarbeitet_oa.csv wurde erfolgreich gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern von D_verarbeitet_oa.csv: {e}")

if __name__ == "__main__":
    main()
