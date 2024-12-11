import pandas as pd
import re
from langdetect import detect, DetectorFactory, LangDetectException
from deep_translator import GoogleTranslator
import spacy
import csv
from spellchecker import SpellChecker

# Set seed for langdetect to ensure consistency
DetectorFactory.seed = 0
#Initiere den Spellchecker
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
    Führt die folgenden Schritte zur Textvorverarbeitung durch:
    1. Entfernt Links.
    2. Entfernt Hashtags.
    3. Ersetzt Satzzeichen gemäß den Vorgaben.
    4. Entfernt Inhalte innerhalb von Klammern und Anführungszeichen.
    5. Entfernt Enter, Tabs, Zahlen und redundante Satzzeichen.
    """
    if pd.isnull(text):
        return ""

    # 1. Entferne Links
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # 2. Entferne Hashtags
    text = re.sub(r'#\w+', '', text)

    detect_and_translate(text)

    # 3. Ersetze ?! und ähnliche durch Punkt
    text = re.sub(r'[?!]+', '.', text)

    # 4. Entferne alle anderen Satzzeichen außer Punkt
    # Ersetze alle Zeichen, die keine Wortzeichen, Leerzeichen oder Punkte sind, durch nichts
    text = re.sub(r'[^\w\s\.,\[\]@]','', text)

    # 5. Entferne Inhalte innerhalb von "" oder ()
    # Falls keine schließende Klammer oder Anführungszeichen, entferne bis zum nächsten Punkt
    def remove_within(text, open_char, close_char):
        # Escape die Zeichen für den regulären Ausdruck
        open_esc = re.escape(open_char)
        close_esc = re.escape(close_char)
        # Regex-Muster zur Suche nach dem offenen Zeichen bis zum schließenden Zeichen oder Punkt
        pattern = re.compile(rf'{open_esc}[^{close_char}\.]*?({close_esc}|\.)')
        while True:
            match = pattern.search(text)
            if not match:
                break
            end = match.end()
            # Ersetze den gefundenen Bereich durch einen Punkt, falls nicht bereits vorhanden
            if match.group(1) == '.':
                replacement = '.'
            else:
                replacement = ''
            text = text[:match.start()] + replacement + text[end:]
        return text

    text = remove_within(text, '"', '"')
    text = remove_within(text, '(', ')')

    # 6. Entferne Enter und Tabs
    text = text.replace('\n', ' ').replace('\t', ' ')

    # 7. Entferne Zahlen
    text = re.sub(r'\d+', '', text)

    # 8. Ersetze Muster wie '. .' oder '... ' durch einen einzigen Punkt
    # Entferne Punkte, die durch Leerzeichen getrennt sind
    text = re.sub(r'\.\s*\.', '.', text)
    # Ersetze mehrfache Punkte durch einen einzigen Punkt
    text = re.sub(r'\.{2,}', '.', text)

    # 9. Entferne überflüssige Leerzeichen
    text = re.sub(r'\s+', ' ', text).strip()

    # 10. Nur Kleinschreibung
    text = text.lower()

    return text


def detect_and_translate(text):
    """
    Erkennt die Sprache des Textes. Wenn es nicht Deutsch ist, übersetzt es ins Deutsche.
    """
    if not text:
        return text
    try:
        lang = detect(text)
    except LangDetectException:
        # Falls die Sprache nicht erkannt werden kann, gehe davon aus, dass es Deutsch ist
        lang = 'de'

    if lang != 'de':
        try:
            translated = GoogleTranslator(source='auto', target='de').translate(text)
            return translated
        except Exception as e:
            print(f"Übersetzungsfehler: {e}")
            return text  # Rückfall auf Originaltext, falls Übersetzung fehlschlägt
    else:

        words = text.split()  # Split the sentence into words
        misspelled = spell.unknown(words)  # Find all misspelled words in one go
        corrections = {word: spell.correction(word) for word in misspelled}  # Get corrections for all misspelled words

        # Replace misspelled words with their corrections
        corrected_words = [corrections[word] if word in corrections and corrections[word] is not None else word for word in words]
        return ' '.join(corrected_words)

def remove_adjectives(text):
    """
    Entfernt alle Adjektive aus dem Text.
    """
    if not text:
        return text
    doc = nlp(text)
    tokens = [token.text for token in doc if token.pos_ != 'ADJ']
    return ' '.join(tokens)

def main():
    print("Beginne mit der Vorverarbeitung der Beschreibungen...")

    # 1. Lies die Datei Data.csv ein
    try:
        df = pd.read_csv('Data/Tweets_Original.csv', sep=';', quotechar='"', encoding='utf-8', quoting=csv.QUOTE_ALL)
    except FileNotFoundError:
        print("Die Datei 'Data.csv' wurde nicht gefunden.")
        return
    except Exception as e:
        print(f"Fehler beim Einlesen der Datei: {e}")
        return

    # Überprüfe, ob die notwendigen Spalten vorhanden sind
    if not {'id', 'description', 'TAR'}.issubset(df.columns):
        print("Die Eingabedatei muss die Spalten 'id', 'description' und 'TAR' enthalten.")
        return

    # 2. Vorverarbeitung der Beschreibung
    print("Starte die Textvorverarbeitung...")
    df['clean_description'] = df['description'].apply(preprocess_text)


    # 4. Speichere alles in D_verarbeitet.csv ab
    df_verarbeitet = df[['id', 'translated_description', 'TAR']].rename(columns={'translated_description': 'description'})
    try:
        df_verarbeitet.to_csv('D_verarbeitet.csv', sep=';', quotechar='"', quoting=csv.QUOTE_ALL, index=False, encoding='utf-8')
        print("D_verarbeitet.csv wurde erfolgreich gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern von D_verarbeitet.csv: {e}")

    # 5. Erstelle eine Version ohne Adjektive
    print("Erstelle eine Version ohne Adjektive...")
    df_verarbeitet['description_oa'] = df_verarbeitet['description'].apply(remove_adjectives)
    df_verarbeitet_oa = df_verarbeitet[['id', 'description_oa', 'TAR']].rename(columns={'description_oa': 'description'})
    try:
        df_verarbeitet_oa.to_csv('D_verarbeitet_oa.csv', sep=';', quotechar='"', quoting=csv.QUOTE_ALL, index=False, encoding='utf-8')
        print("D_verarbeitet_oa.csv wurde erfolgreich gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern von D_verarbeitet_oa.csv: {e}")

if __name__ == "__main__":
    main()
