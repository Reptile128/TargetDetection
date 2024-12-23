import pandas as pd
import numpy as np
import spacy
import re
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn_crfsuite import CRF, metrics as crf_metrics
from sklearn.metrics import precision_recall_fscore_support, classification_report
from scipy.sparse import hstack
from gensim.models import Word2Vec
from joblib import Parallel, delayed
import nltk
from nltk.stem.snowball import GermanStemmer

############################################################
# Parameter-Definitionen
############################################################

DATA_PATH = 'Data/Data_clean.csv'  # Pfad zur bereinigten Datendatei
FAMILIENNAMEN_PATH = 'Data/Familiennamen.txt'  # Pfad zur Datei mit Familiennamen
VORNAMEN_PATH = 'Data/Vornamen.txt'  # Pfad zur Datei mit Vornamen

# Arrays zur Erprobung verschiedener Hyperparameter-Kombinationen
TOPN_SIMILAR_VALUES = [0, 5]  # Verschiedene Werte für TOPN_SIMILAR
SINGLE_SENT_WEIGHTS = [1.0]  # Gewichtungen für Ein-Satz-Tweets
MULTI_SENT_WEIGHTS = [0.8]  # Gewichtungen für Mehr-Satz-Tweets
CRF_NGRAM_SIZES = [10, 15]  # n-Gram-Größen für CRF-Features
POS_CONTEXT_LENGTHS = [3]  # Kontextlängen für POS-Features
OVERSAMPLING_FACTORS = [1]  # Oversampling-Faktoren für CRF
CRF_LABELS = ['group', 'individual', 'public']  # Mögliche CRF-Labels für die Vorhersage von den Gruppen
CHECK_NOUN_ART_PRON_ENT = False  # Flag zur Filterung von Tokens
OUTPUT_FILE = 'ergebnisse.txt'  # Pfad zur Ausgabedatei

stemmer = GermanStemmer()  # Initialisierung des deutschen Stemmer

############################################################
# Hilfsfunktionen
############################################################

def read_data(data_path):
    """
    Liest die CSV-Daten ein und prüft auf erforderliche Spalten.

    Args:
        data_path (str): Pfad zur CSV-Datei.
    Returns:
        pd.DataFrame: Eingelesener DataFrame.
    Raises:
        ValueError: Wenn erforderliche Spalten fehlen.
    """
    df = pd.read_csv(data_path, delimiter=';', quotechar='"', encoding='utf-8')
    required_columns = ['id', 'description', 'TAR']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Datei muss Spalten {required_columns} enthalten.")
    df['description'] = df['description'].fillna('').astype(str)
    return df


def sentence_count(text):
    """
    Zählt die Anzahl der Sätze in einem Text.

    Args:
        text (str): Der zu analysierende Text.
    Returns:
        int: Anzahl der Sätze im Text, mindestens 1 (Falls am Ende kein Punkt ist).
    """
    splits = re.split(r'\.+', text)
    count = sum(1 for s in splits if s.strip())
    return max(count, 1)


def load_names(file_path):
    """
    Lädt Namen aus einer Textdatei in ein Set, wenn diese Datei existiert

    Args:
        file_path (str): Pfad zur Namensdatei.
    Returns:
        set: Die Namen kleingeschrieben
    """
    if not os.path.exists(file_path):
        return set()
    with open(file_path, 'r', encoding='utf-8') as f:
        names = {line.strip().lower() for line in f if line.strip()}
    return names


def is_text_empty(desc):
    """
    Überprüft, ob ein Text keine Buchstaben enthält, da er dann zu public gehört. Links wurden zuvor bei der
    Vorverarbeitung entfernt.

    Args:
        desc (str): Der zu prüfende Text.
    Returns:
        bool: True, wenn der Text leer ist, sonst False.
    """
    return not bool(re.search('[a-zA-Z]', desc))


def extract_entity_from_special(token_text):
    """
    Extrahiert Entitäten aus den speziellen Angaben, um diese ebenfalls bei NER zu deklarieren.
    Beispiel: '[@Entität]' wird zu 'Entität'. Dient dazu, dass die @PRE etc auch als Entität erkannt werden.
    Args:
        token_text (str): Der zu prüfende Text
    Returns:
        str: Extrahierte Entität oder 'O', wenn es keine Entität ist.
    """
    m = re.match(r'\[@([^]]+)\]', token_text, re.IGNORECASE)
    if m:
        return m.group(1)
    return 'O'


def process_text(text, nlp, vornamen, nachnamen):
    """
    Verarbeitet den Text mit SpaCy, extrahiert Tokens und deren Merkmale.

    Tokenisiert den Text und bestimmt dann anschließend für jedes Token den POS-TAG, ob es ein Name ist und/oder Entität
    (inkl. der speziellen Tokens [@IND])

    Args:
        text (str): Der zu verarbeitende Text.
        nlp (spacy.lang.de.DeGerman): Das deutsche SpaCy NLP-Modell.
        vornamen (set): Die zuvor geladenen Vornamen.
        nachnamen (set): Die zuvor geladenen Nachnamen.

    Returns:
        tuple: Das SpaCy-Doc-Objekt und eine Liste von Token-Tupeln.
    """

    doc = nlp(text)
    tokens = []
    for t in doc:
        if t.is_space or t.text == ".":         # Überspringe Leerzeichen und Punkte
            continue
        token_text = t.text.lower()
        token_pos = t.pos_                      # Bestimme TAG-POS des Tokens
        is_name = (token_text in vornamen or token_text in nachnamen)       # Überprüfe, ob es in Namen ist.
        ent_type = t.ent_type_ if t.ent_type_ else 'O'                      # Überprüfe, ob es ein Entity ist
        if re.match(r'\[@[^]]+\]', t.text):                                 # Überprüfe, ob es ein Special-Entity ist
            ent_type = extract_entity_from_special(t.text)
        tokens.append((token_text, token_pos, is_name, ent_type))
    return doc, tokens


def vectorize_and_weight(texts, sentence_counts, single_weight, multi_weight):
    """
    Vektorisiert Texte mit TF-IDF und gewichtet die Zeilen basierend auf Satzanzahl.
    Einzelne Sätze sind eindeutiger bzgl. wer angesprochen wird. Dementsprechend findet dort eine Gewichtung statt.

    Args:
        texts (list): Liste der zu vektorisierenden Texte.
        sentence_counts (array-like): Anzahl der Sätze pro Text.
        single_weight (float): Gewicht für Ein-Satz-Tweets.
        multi_weight (float): Gewicht für Mehr-Satz-Tweets.

    Returns:
        tuple: Vektorisierter und gewichteter TF-IDF-Matrix sowie das TF-IDF-Modell.
    """
    tfidf = TfidfVectorizer()           # Initialisiert TF-IDF-Vektorisierer, um TF-IDF zu erstellen
    X = tfidf.fit_transform(texts)      # Passt den Vektorisierer an die Eingabetexte an
    row_weights = np.array([single_weight if c == 1 else multi_weight for c in sentence_counts])    # Weißt den Daten eine Gewichtung zu, je nach Satzanzahl
    X = X.multiply(row_weights[:, None])        # Gewichtet die Daten unterschiedlich stark. [:,None] notwendig um richtiges Format zu haben
    return X, tfidf


def transform_and_weight(tfidf, texts, sentence_counts, single_weight, multi_weight):
    """
    Transformiert und gewichtet neue Texte mit einem bereits trainierten TF-IDF-Modell.

    Args:
        tfidf (TfidfVectorizer): Vorgefülltes TF-IDF-Modell.
        texts (list): Liste der zu transformierenden Texte.
        sentence_counts (array-like): Anzahl der Sätze pro Text.
        single_weight (float): Gewicht für Ein-Satz-Tweets.
        multi_weight (float): Gewicht für Mehr-Satz-Tweets.

    Returns:
        sparse matrix: Gewichtete TF-IDF-Matrix.
    """
    X = tfidf.transform(texts)                  # Transformiert die neuen Texte in die TF-IDF-Matrix
    row_weights = np.array([single_weight if c == 1 else multi_weight for c in sentence_counts])    # Weißt den Daten eine Gewichtung zu, je nach Satzanzahl
    X = X.multiply(row_weights[:, None])        # Multipliziert die Daten [:,None] notwendig um richtiges Format zu haben
    return X


def get_best_tfidf_similar_word(model_w2v, word, idf_all, feature_names_all, top_n_similar):
    """
    Findet die ähnlichsten Wörter basierend auf Word2Vec bei Nomen, die im TF-IDF-Vokabular vorhanden sind.
    Dies soll helfen, unbekannte Wörter besser einordnen zu können, indem mehrere relevante ähnliche Wörter zurückgegeben werden.

    Args:
        model_w2v (Word2Vec): Trainiertes Word2Vec-Modell.
        word (str): Das Ausgangswort.
        idf_all (array-like): IDF-Werte für alle TF-IDF-Features.
        feature_names_all (array-like): Namen aller TF-IDF-Features.
        top_n_similar (int): Anzahl der ähnlichen Wörter, die vom Word2Vec-Modell abgerufen werden sollen. Gibt diese dann zurück, falls sie in TF-IDF sind

    Returns:
        list: Liste der ähnlichsten Wörter mit den höchsten IDF-Werten oder eine leere Liste, falls keine passenden Wörter gefunden werden.
    """

    if word not in model_w2v.wv:        # Überprüfe, ob das Ausgangswort im Word2Vec-Vokabular vorhanden ist
        return []                       # Wenn das Wort nicht im Vokabular ist, gebe eine leere Liste zurück

    # Hole die top_n_similar ähnlichsten Wörter basierend auf dem Word2Vec-Modell
    similar = model_w2v.wv.most_similar(word, topn=top_n_similar)

    best_words = []  # Liste zur Speicherung der besten Wörter
    best_idfs = []   # Liste zur Speicherung der entsprechenden IDF-Werte


    vocab_map = {f: i for i, f in enumerate(feature_names_all)} # Mache ein Dictionary daraus für den vereinfachten Zugriff

    for w_sim, _ in similar: # Iteriere über die ähnlichsten Wörter und finde die besten basierend auf dem IDF-Wert
        w_sim_lower = w_sim.lower()

        if w_sim_lower in vocab_map:        # Überprüfe, ob das ähnliche Wort im TF-IDF-Vokabular vorhanden ist
            idx = vocab_map[w_sim_lower]    # Hole den Index des Wortes im TF-IDF-Vokabular
            w_idf = idf_all[idx]            # Rufe den IDF-Wert des Wortes ab

            best_words.append(w_sim_lower)  # Füge das Wort hinzu
            best_idfs.append(w_idf)         # Füge den IDF Wert hinzu

    # Verbinde die Wörter mit ihren IDF-Werten und sortiere sie nach Relevanz (IDF-Wert)
    sorted_words_idfs = sorted(zip(best_words, best_idfs), key=lambda x: x[1], reverse=True)

    return sorted_words_idfs


def get_word2vec_features_nouns(model_w2v, w, idf_all, feature_names_all, p, top_n_similar):
    """
    Extrahiert Word2Vec-Features für Nomen.

    Diese Funktion verwendet ein Word2Vec-Modell, um ähnliche Wörter für ein gegebenes Nomen zu finden,
    die auch im TF-IDF-Vokabular vorhanden sind. Die ähnlichsten Wörter mit den höchsten IDF-Werten
    werden als Features zurückgegeben, um die Bedeutung und Relevanz des Nomens im Kontext zu verstärken.

    Args:
        model_w2v (Word2Vec): Trainiertes Word2Vec-Modell.
        w (str): Das Wort (Nomen), für das ähnliche Wörter gefunden werden sollen.
        idf_all (array-like): IDF-Werte für alle TF-IDF-Features.
        feature_names_all (array-like): Namen aller TF-IDF-Features.
        p (str): Part-of-Speech-Tag des Wortes.
        top_n_similar (int): Anzahl der ähnlichen Wörter für Word2Vec.

    Returns:
        dict: Dictionary mit den besten ähnlichen TF-IDF-Wörtern oder leer, wenn das Wort kein Nomen ist oder keine ähnlichen Wörter gefunden wurden.
    """
    # Überprüfe, ob das Wort ein Nomen ist
    if p == 'NOUN':
        # Finde die besten ähnlichen TF-IDF-Wörter basierend auf dem Word2Vec-Modell
        best_tfidf_words = get_best_tfidf_similar_word(
            model_w2v, w, idf_all, feature_names_all, top_n_similar
        )

        # Wenn es nicht leer ist, füge es den Features hinzu
        if best_tfidf_words:
            return {'best_sim_tfidf_words': best_tfidf_words}

    # Falls leer, gebe leere Liste zurück
    return {}


def stem_noun(w):
    """
    Stemmt ein Nomen und ermittelt die Differenz zwischen Stamm und Originalwort (maximal 2 Zeichen).
    Dadurch sollen Endungen erkannt werden.

    Args:
        w (str): Das Nomen.

    Returns:
        tuple: Stamm des Nomens und Differenz am Ende des Wortes (maximal 3 Zeichen).
    """
    stem = stemmer.stem(w)      # Stemmt das Nomen
    if w.startswith(stem):      # Überprüft, ob Stamm gleich geblieben ist
        diff = w[len(stem):]    # Extrahiere die Endung (Differenz) nach dem Stamm
    else:                       # Leerer String, falls unterschiedlich
        diff = ''

    diff = diff[:2]             # Begrenze auf 2 Zeichen, da nur diese relevant sind
    return stem, diff


def stem_adj(w):
    """
    Stemmt ein Adjektiv. Gibt dann die Differenz (max. 2 Zeichen) zurück.

    Args:
        w (str): Das Adjektiv.

    Returns:
        str: Die Differenz zwischen Stamm und Originalwort.
    """
    stem = stemmer.stem(w)
    if w.startswith(stem):
        # Extrahiere die Endung (Differenz) nach dem Stamm
        diff = w[len(stem):]
        diff = diff[:2]  # Begrenze die Differenz auf maximal 2 Zeichen
    else:
        # Falls das Wort nicht mit dem Stamm übereinstimmt, nimm die letzten 2 Zeichen
        diff = w[-2:]

    return diff


def extract_np_spans(doc):
    """
    Extrahiert Nominalphrasen (NP) aus dem SpaCy-Dokument, sowie Pronomen und Artikeln, welche nicht Teil von NP's sind.

    Args:
        doc (spacy.tokens.Doc): Das SpaCy-Dokument.

    Returns:
        list: Liste von NPs, wobei jede NP eine Liste aus den Tupeln.
    """
    nps = []
    # Extrahiert NPs aus den Noun-Chunks von SpaCy
    for chunk in doc.noun_chunks:       # Iteriere durch alle NP's chunks, welche von Spacy erkannt wurden
        np_tokens = []
        # Füge jedes Token der NP hinzu und speichere zusätzlich POS-TAG sowie NER
        for t in chunk:
            np_tokens.append((t.text.lower(), t.pos_, t.ent_type_ if t.ent_type_ else 'O'))
        nps.append(np_tokens)

    # Erstelle ein Set für bereits extrahierte für den nachfolgenden Schritt
    existing_np_words = set(w for np_chunk in nps for w, p, e in np_chunk)

    # Füge die Pronomen und Artikel hinzu, die nicht bereits in den Noun-Chunks enthalten sind
    for t in doc:
        w = t.text.lower()
        p = t.pos_
        e = t.ent_type_ if t.ent_type_ else 'O'
        if (p in ['PRON', 'DET']) and w not in existing_np_words:
            nps.append([(w, p, e)])
            existing_np_words.add(w)

    return nps


def filter_tokens_for_crf(doc_tokens, doc):
    """
    Filtert Tokens für die Verwendung im CRF-Modell basierend auf Nominalphrasen und POS-Tags.

    Args:
        doc_tokens (list): Liste von Token-Tupeln (w, p, isn, ent).
        doc (spacy.tokens.Doc): Das SpaCy-Dokument.

    Returns:
        list: Gefilterte Liste von Token-Tupeln.
    """
    # Extrahiere NP's, Artikel und Pronomen aus dem Dokument
    nps = extract_np_spans(doc)
    np_words = set(ww for np_chunk in nps for ww, pp, ee in np_chunk)   # Erstellt Menge aller Wörter, die Teil einer NP sind

    filtered = []
    for (w, p, isn, ent) in doc_tokens:
        if w == '.':
            break  # Stoppt die Filterung bei einem Punkt, um es für die Sätze zu machen
        in_np = w in np_words
        if in_np:
            filtered.append((w, p, isn, ent))  # NPs immer hinzufügen
        else:
            if CHECK_NOUN_ART_PRON_ENT:
                # Hinzufügen, wenn POS in [NOUN, DET, PRON] oder Entität vorhanden (nur wenn auf TRUE gesetzt ist)
                if (p in ['NOUN', 'DET', 'PRON'] or ent != 'O'):
                    filtered.append((w, p, isn, ent))
    return filtered


def ngram_tokens_overlapping(doc_tokens, n):
    """
    Erzeugt n-Gramme von Tokens mit Überlappung.

    Args:
        doc_tokens (list): Liste von Token-Tupeln (w, p, isn, ent).
        n (int): Die gewünschte Größe des n-Gramms.

    Returns:
        list: Liste von n-Gram-Token-Tupeln.
    """
    merged = []
    # Iteriere über die Token Liste und erstelle n-Gramme mit Überlappung
    for i in range(len(doc_tokens) - n + 1):
        chunk = doc_tokens[i:i + n]
        w_merged = "_".join([x[0] for x in chunk])  # Setze die Wörter mit _ zusammen
        p_first = chunk[0][1]  # Überprüfe POS des ersten Tokens
        is_name_any = any(x[2] for x in chunk)  # Prüft, ob eines der Tokens ein Name ist
        ent_first = chunk[0][3]  # Entität des ersten Tokens
        merged.append((w_merged, p_first, is_name_any, ent_first))  # Füge es in die Liste hinzu
    return merged


def ner_boost_pos_text(doc_tokens):
    """
    Verstärkt Wörter, welche mittels NER als Entity erkannt wurden (oder special Entities sind)

    Args:
        doc_tokens (list): Liste von Token-Tupeln (w, p, isn, ent).

    Returns:
        str: Verstärkter Text basierend auf POS-Tags und Entitäten.
    """
    words = []
    for w, p, isn, ent in doc_tokens:   # Iteriere durch die Tokens
        if p in ['NOUN', 'DET', 'PRON']:    # Nur für Nomen/Det/Pronomen
            if ent != 'O':                  # Wenn es eine erkannte Entity ist, dann füge es doppelt hinzu
                words.extend([w, w])  # Wiederhole Wörter mit Entitäten
            else:
                words.append(w)             # Rest soll normal hinzugefügt werden
    return ' '.join(words)


def join_tokens_all(doc_tokens):
    """
    Fügt alle Tokens zu einem zusammenhängenden String zusammen.

    Args:
        doc_tokens (list): Liste von Token-Tupeln (w, p, isn, ent).

    Returns:
        str: Zusammengesetzter Text.
    """
    return ' '.join([w for w, p, isn, ent in doc_tokens])


def one_hot_label(lab, crf_labels):
    """
    Wandelt ein Label in ein One-Hot-Encoded-Format um, sodass der Algorithmus damit umgehen kann

    Args:
        lab (str): Das zu kodierende Label.
        crf_labels (list): Liste der möglichen Labels, welche oben deklariert wurden

    Returns:
        list: One-Hot-Encoded Liste.
    """
    return [1 if lab == c else 0 for c in crf_labels]


def oversample_for_crf(X_crf, Y_crf, sentence_counts, single_weight, multi_weight, oversampling_factor):
    """
    Führt Oversampling für das CRF-Modell durch, um Klassenungleichgewichte zu adressieren.

    Args:
        X_crf (list): Liste der CRF-Features.
        Y_crf (list): Liste der CRF-Labels.
        sentence_counts (array-like): Anzahl der Sätze pro Beispiel.
        single_weight (float): Gewicht für Ein-Satz-Tweets.
        multi_weight (float): Gewicht für Mehr-Satz-Tweets.
        oversampling_factor (int): Faktor zur Skalierung des Oversamplings.

    Returns:
        tuple: Neue überabgestimmte Listen von CRF-Features und Labels (Gibt neue Liste der CRF-Features/-Labels zurück).
    """
    # Berechnung der Wiederholungen für Ein- und Mehrsatz Beispiele. Muss mind. 1 sein.
    single_rep = max(1, int(round(single_weight * oversampling_factor)))
    multi_rep = max(1, int(round(multi_weight * oversampling_factor)))

    X_new = []
    Y_new = []
    # Iteriere durch die ursprünglichen Daten
    for x_seq, y_seq, scount in zip(X_crf, Y_crf, sentence_counts):
        # Prüfe, ob ein oder mehrere Sätze.
        if scount == 1:
            for _ in range(single_rep):     # Füge es >1x hinzu (abhängig von Gewichtung)
                X_new.append(x_seq)         # Füge die Sequenz hinzu
                Y_new.append(y_seq)         # Füge das dazugehörige Label hinzu
        else:                               # Füge es bei mehreren Sätzen >1x hinzu
            for _ in range(multi_rep):
                X_new.append(x_seq)
                Y_new.append(y_seq)
    return X_new, Y_new                     # Gibt die Daten mit angepasster Gewichtung hinzu


def get_context_features(w_index, sent_toks, pos_context_length):
    """
    Extrahiert vorherige und nachfolgende Entitäten und POS-TAG-N für das CRF

    Args:
        w_index (int): Index des aktuellen Wortes in der Satzliste.
        sent_toks (list): Liste von Token-Tupeln im Satz.
        pos_context_length (int): Länge des POS-Kontexts.

    Returns:
        dict: Kontext-Features.
    """
    features = {}
    # Hole die oben bestimmten X Wörter davor/danach
    for offset in range(1, pos_context_length + 1):
        # Kontext vor dem aktuellen Wort
        if w_index - offset >= 0:       # Stelle sicher, dass der Index gültig bleibt
            pw, pp, pe = sent_toks[w_index - offset]    # Hole Token Tupel vor dem aktuellen Wort
            features[f'-{offset}:pos'] = pp             # Speichere POS-TAG
            features[f'-{offset}:ent'] = str(pe != 'O') # Speichere Entity
        # Kontext nach dem aktuellen Wort
        if w_index + offset < len(sent_toks):           # Nochmal dasselbe nur für Wörter danach
            nw, np_, ne = sent_toks[w_index + offset]
            features[f'+{offset}:pos'] = np_
            features[f'+{offset}:ent'] = str(ne != 'O')
    return features


def extract_crf_features_for_doc(doc_tokens, doc, model_w2v, idf_all, feature_names_all, pos_context_length, vornamen,
                                 nachnamen, top_n_similar):
    """
    Extrahiert die Features für das CRF-Modell aus einem Dokument. Tokens (Entität, Wortart), Nominalphrasen,
    Kontext (umliegende Wörter) und word2vec

    Args:
        doc_tokens (list): Liste der gefilterten Token-Tupeln für das CRF.
        doc (spacy.tokens.Doc): Das SpaCy-Dokument.
        model_w2v (Word2Vec): Trainiertes Word2Vec-Modell.
        idf_all (array-like): IDF-Werte der TF-IDF-Features.
        feature_names_all (array-like): Namen der TF-IDF-Features.
        pos_context_length (int): Länge des POS-Kontexts.
        vornamen (set): Menge der Vornamen.
        nachnamen (set): Menge der Nachnamen.
        top_n_similar (int): Anzahl der ähnlichen Wörter für Word2Vec-Features.

    Returns:
        list: Liste der Feature-Dictionaries für jedes Token.
    """
    # Extrahiert die NP's aus dem Dokument und fügt sie token2np hinzu
    nps = extract_np_spans(doc)
    token2np = {}
    for np_chunk in nps:
        for ww, pp, ee in np_chunk:
            token2np[ww] = np_chunk

    # Aufteilung der Tweets in einzelne Sätze.
    saetze = list(doc.sents)         # Teilt den Text in Sätze auf
    satz_tokens = []
    tweet_linear_tokens = []              # Liste aller Tokens (Wort, POS, Entität)
    for satz in saetze:
        stoks = []
        for tt in satz:
            wort_ = tt.text.lower()
            pos_ = tt.pos_
            entity_ = tt.ent_type_ if tt.ent_type_ else 'O'         # Falls es keine Entity ist mache es O
            stoks.append((wort_, pos_, entity_))
            tweet_linear_tokens.append((wort_, pos_, entity_))
        satz_tokens.append(stoks)

    # Finden der Übereinstimmungen zwischen tweet_tokens und den Satz-Tokens
    matched_indices = []
    used = [False] * len(tweet_linear_tokens)
    for idx, (w, p, isn, ent) in enumerate(doc_tokens):
        for j, (ww, wp, we) in enumerate(tweet_linear_tokens):
            if not used[j] and ww == w:
                satz_id = 0
                c = 0
                for si, st in enumerate(satz_tokens):
                    if j < c + len(st):
                        satz_id = si
                        wort_id = j - c
                        break
                    c += len(st)
                matched_indices.append((idx, satz_id, wort_id))
                used[j] = True
                break

    features_list = []
    # Iteriere durch die Tokens und extrahiere die Features
    for idx, (w, p, isn, ent) in enumerate(doc_tokens):
        # Basis-Feature für jedes Token
        token_features = {
            'bias': 1.0,  # Bias-Feature
            'word.lower()': w,  # Kleinbuchstaben des Wortes
            'pos': p,  # POS-Tag
            'is_name': str(isn),  # Flag, ob es ein Name ist
            'ent_type': ent  # Entitätstyp
        }

        # Prüfe, ob das Token ein NP ist und extrahiere die Merkmale
        if w in token2np:
            np_tokens = token2np[w]
            # Extrahiere Artikel
            det_words = [x[0] for x in np_tokens if x[1] == 'DET']
            if det_words:
                token_features['np_det'] = '_'.join(det_words)

            # Extrahiere und verwende Differenz des Adjektivs
            adj_tokens = [x[0] for x in np_tokens if x[1] == 'ADJ']
            if adj_tokens:
                stemmed_adjs = [stem_adj(a) for a in adj_tokens]
                token_features['np_adjs_stemmed'] = '_'.join(stemmed_adjs)

            # Extrahiere und stemme Nomen. Verwende Differenz. Mache das für alle Nomen
            noun_tokens = [x[0] for x in np_tokens if x[1] == 'NOUN']
            if noun_tokens:
                stems = []
                diffs = []
                for no in noun_tokens:
                    s_n, diff_n = stem_noun(no)
                    stems.append(s_n)
                    if diff_n:
                        diffs.append(diff_n)
                token_features['np_noun_stem'] = '_'.join(stems)
                if diffs:
                    token_features['np_noun_diff'] = '_'.join(diffs)

            # Extrahiere Pronomen
            pron_tokens = [x[0] for x in np_tokens if x[1] == 'PRON']
            if pron_tokens:
                token_features['np_pron'] = '_'.join(pron_tokens)

            # Überprüfe, ob die NP einen Namen enthält
            if any(x[0] in vornamen or x[0] in nachnamen for x in np_tokens):
                token_features['np_has_name'] = True

        # Word2Vec-Feature um semantisch ähnliche Wörter hinzuzufügen.
        if model_w2v is not None and idf_all is not None and feature_names_all is not None:
            w2v_feats = get_word2vec_features_nouns(model_w2v, w, idf_all, feature_names_all, p, top_n_similar)
            token_features.update(w2v_feats)

        # Kontext-Features um Informationen über benachbarte Wörter zu erhalten
        satz_id = None      # Initialisierung der Satz-ID und Wort-ID
        wort_id = None
        #  Überprüfe, ob der Index des aktuellen Tokens (idx) in den matched_indices ist
        for m in matched_indices:
            if m[0] == idx:
                satz_id = m[1]
                wort_id = m[2]
                break # Wenn gefunden, beende die Suche
        # Wenn gefunden, extrahiere die Kontext-Features
        if satz_id is not None and wort_id is not None:
            satz_toks = satz_tokens[satz_id] # Hole die Tokens des Satzes
            c_feats = get_context_features(wort_id, satz_toks, pos_context_length) # Extrahiere die Kontext-Features (berücksichtigt pos_context_length)
            token_features.update(c_feats) # Füge die Kontext-Features hinzu

        features_list.append(token_features)

    return features_list


def run_experiment(SINGLE_SENT_WEIGHT, MULTI_SENT_WEIGHT, NGRAM_SIZE, POS_CONTEXT_LEN,
                   top_n_similar, oversampling_factor,
                   train_df, test_df, crf_labels):
    """
    Führt ein komplettes Experiment mit den angegebenen Hyperparametern durch.

    Args:
        SINGLE_SENT_WEIGHT (float): Gewichtung für Ein-Satz-Tweets.
        MULTI_SENT_WEIGHT (float): Gewichtung für Mehr-Satz-Tweets.
        NGRAM_SIZE (int): Größe der n-Gramme für CRF-Features.
        POS_CONTEXT_LEN (int): Kontextlänge für POS-Features.
        top_n_similar (int): Anzahl der ähnlichen Wörter für Word2Vec.
        oversampling_factor (int): Faktor für das Oversampling beim CRF.
        train_df (pd.DataFrame): Trainingsdaten.
        test_df (pd.DataFrame): Testdaten.
        crf_labels (list): Liste der möglichen CRF-Labels.

    Returns:
        dict: Dictionary mit den Ergebnissen des Experiments.
    """
    # Laden von Vornamen und Nachnamen
    vornamen = load_names(VORNAMEN_PATH)
    nachnamen = load_names(FAMILIENNAMEN_PATH)

    # Laden des SpaCy-Modells
    nlp = spacy.load('de_core_news_sm')

    # Überprüfen, ob die Beschreibung leer ist
    train_df['is_empty'] = train_df['description'].apply(is_text_empty)
    test_df['is_empty'] = test_df['description'].apply(is_text_empty)

    # Verarbeiten des Textes mit SpaCy
    # Wendet die Funktion 'process_text' auf jede Beschreibung an und speichert das Ergebnis als 'doc_tuple'
    train_df['doc_tuple'] = train_df['description'].apply(lambda x: process_text(x, nlp, vornamen, nachnamen))
    # Extrahiert das SpaCy-Doc-Objekt aus 'doc_tuple'
    train_df['doc'] = train_df['doc_tuple'].apply(lambda x: x[0])
    # Extrahiert die Token-Tupel aus dem 'doc_tuple'
    train_df['doc_tokens'] = train_df['doc_tuple'].apply(lambda x: x[1])
    # Entfernt die temporäre Spalte 'doc_tuple'
    train_df.drop(columns=['doc_tuple'], inplace=True)

    # Wiederhole den Vorgang für die Testdaten
    test_df['doc_tuple'] = test_df['description'].apply(lambda x: process_text(x, nlp, vornamen, nachnamen))
    test_df['doc'] = test_df['doc_tuple'].apply(lambda x: x[0])
    test_df['doc_tokens'] = test_df['doc_tuple'].apply(lambda x: x[1])
    test_df.drop(columns=['doc_tuple'], inplace=True)

    # Filtern der Tokens für das CRF-Modell
    train_df['doc_tokens_crf'] = train_df.apply(lambda row: filter_tokens_for_crf(row['doc_tokens'], row['doc']), axis=1)
    test_df['doc_tokens_crf'] = test_df.apply(lambda row: filter_tokens_for_crf(row['doc_tokens'], row['doc']), axis=1)

    # Erzeugen von n-Grammen, welche sich überlappen
    train_df['doc_tokens_crf_ng'] = train_df['doc_tokens_crf'].apply(lambda dt: ngram_tokens_overlapping(dt, NGRAM_SIZE))
    test_df['doc_tokens_crf_ng'] = test_df['doc_tokens_crf'].apply(lambda dt: ngram_tokens_overlapping(dt, NGRAM_SIZE))

    # Trainieren des Word2Vec-Modells mit den eigenen Tweets
    sentences = [[w for w, p, isn, ent in doc] for doc in train_df['doc_tokens_crf_ng']]
    model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=20)

    # Erzeugen von TF-IDF-Texten
    # Verstärkt die Texte basierend auf POS-Tags und Entitäten, um zusätzliche syntaktische Informationen zu integrieren
    train_texts_all = train_df['doc_tokens'].apply(join_tokens_all).tolist()
    test_texts_all = test_df['doc_tokens'].apply(join_tokens_all).tolist()

    # Erzeugen von POS-gestärkten Texten
    train_texts_pos = train_df['doc_tokens'].apply(ner_boost_pos_text).tolist()
    test_texts_pos = test_df['doc_tokens'].apply(ner_boost_pos_text).tolist()

    # Extrahieren der Satzanzahlen
    train_sentence_counts = train_df['sentence_count'].values
    test_sentence_counts = test_df['sentence_count'].values

    # Vektorisieren und Gewichtung der TF-IDF-Features. Anwendung der Gewichtung basierend auf der Satzanzahl
    X_train_all, tfidf_all = vectorize_and_weight(train_texts_all, train_sentence_counts, SINGLE_SENT_WEIGHT, MULTI_SENT_WEIGHT)
    X_test_all = transform_and_weight(tfidf_all, test_texts_all, test_sentence_counts, SINGLE_SENT_WEIGHT, MULTI_SENT_WEIGHT)

    # Vektorisieren und Gewichtung der POS-TF-IDF-Features wie beim TF-IDF zuvor.
    X_train_pos, tfidf_pos = vectorize_and_weight(train_texts_pos, train_sentence_counts, SINGLE_SENT_WEIGHT, MULTI_SENT_WEIGHT)
    X_test_pos = transform_and_weight(tfidf_pos, test_texts_pos, test_sentence_counts, SINGLE_SENT_WEIGHT, MULTI_SENT_WEIGHT)

    # Extrahieren der Feature-Namen und IDF-Werte aus dem TF-IDF-Modell
    feature_names_all = np.array(tfidf_all.get_feature_names_out())
    idf_all = tfidf_all.idf_  # Holt die IDF-Werte für jedes TF-IDF-Feature

    # Extrahieren von CRF-Features und Labels für den Trainingsdatensatz
    X_train_crf = []
    y_train_crf = []

    # Hole alle CRF-Features und Labels für den Trainingsdatensatz
    for i, row in train_df.iterrows():
        feats = extract_crf_features_for_doc(
            row['doc_tokens_crf_ng'], row['doc'], model_w2v, idf_all,
            feature_names_all, POS_CONTEXT_LEN, vornamen, nachnamen, top_n_similar
        )
        # Erstellt die Labels für jeden Token im Dokument
        labels = [row['TAR']] * len(row['doc_tokens_crf_ng'])  # Label für jedes Token
        X_train_crf.append(feats)   # Füge die Features hinzu
        y_train_crf.append(labels)  # Füge die Labels hinzu

    # Dasselbe für den Testdatensatz
    X_test_crf = []
    y_test_crf = []
    for i, row in test_df.iterrows():
        feats = extract_crf_features_for_doc(
            row['doc_tokens_crf_ng'], row['doc'], model_w2v, idf_all,
            feature_names_all, POS_CONTEXT_LEN, vornamen, nachnamen, top_n_similar
        )
        labels = [row['TAR']] * len(row['doc_tokens_crf_ng'])
        X_test_crf.append(feats)
        y_test_crf.append(labels)

    # Oversampling zur Handhabung von Klassenungleichgewichten
    X_train_crf_ov, y_train_crf_ov = oversample_for_crf(
        X_train_crf, y_train_crf, train_sentence_counts,
        SINGLE_SENT_WEIGHT, MULTI_SENT_WEIGHT, oversampling_factor
    )

    # Trainieren des CRF-Modells
    crf = CRF(
        algorithm='lbfgs', c1=0.1, c2=0.1,  # c1=0.1 hilft überflüssige Parameter auf Null zu setzen, c2=0.1 hilft um Overfitting zu verhindern
        max_iterations=200, all_possible_transitions=True   # max_iterations=200 hilft um das Modell zu trainieren
    )
    crf.fit(X_train_crf_ov, y_train_crf_ov) # Trainiere das CRF-Modell mit den oversampelten Daten

    # Vorhersagen mit dem CRF-Modell
    y_pred_crf = crf.predict(X_test_crf)    # Führt die Vorhersagen mit den Testdaten durch
    # Extrahiere die Labels für die Vorhersagen oder setze
    y_pred_crf_tweet = [tokens[0] if len(tokens) > 0 else 'public' for tokens in y_pred_crf]
    crf_report = crf_metrics.flat_classification_report(y_test_crf, y_pred_crf, digits=4)

    # Erzeugen von Word2Vec-Embeddings für die Dokumente. Berechnung des Durchschnitts der Word2Vec-Embeddings
    train_w2v_emb = np.array([
        np.mean(
            [model_w2v.wv[w] for w, p, i, e in doc if w in model_w2v.wv] or [np.zeros(model_w2v.wv.vector_size)],
            axis=0
        )
        for doc in train_df['doc_tokens_crf_ng']
    ])
    # Selbiges für den Testdatensatz
    test_w2v_emb = np.array([
        np.mean(
            [model_w2v.wv[w] for w, p, i, e in doc if w in model_w2v.wv] or [np.zeros(model_w2v.wv.vector_size)],
            axis=0
        )
        for doc in test_df['doc_tokens_crf_ng']
    ])

    # One-Hot-Encoding der CRF-Labels (Umwandeln in Vektoren, sodass das Modell damit umgehen kann)
    y_train_crf_tweet = [t[0] if len(t) > 0 else 'public' for t in y_train_crf]

    def ohe(l): # Hilfsfunktion für das One-Hot-Encoding
        return one_hot_label(l, crf_labels)

    # One-Hot-Encoding für Trainings- und Testdaten
    train_crf_ohe = np.array([ohe(l) for l in y_train_crf_tweet])
    test_crf_ohe = np.array([ohe(l) for l in y_pred_crf_tweet])

    # Extrahieren der Wortanzahlen
    train_word_counts = np.array([len(doc) for doc in train_df['doc_tokens']]).reshape(-1, 1)
    test_word_counts = np.array([len(doc) for doc in test_df['doc_tokens']]).reshape(-1, 1)

    # Extrahieren des leeren Text-Flags
    train_empty_feat = train_df['is_empty'].astype(int).values.reshape(-1, 1)
    test_empty_feat = test_df['is_empty'].astype(int).values.reshape(-1, 1)

    # Kombinieren der TF-IDF-Features
    X_train_tfidf_combined = hstack([X_train_all, X_train_pos]).tocsr()
    X_test_tfidf_combined = hstack([X_test_all, X_test_pos]).tocsr()

    # Kombinieren aller Features zu finalen Trainings- und Test-Matrizen
    X_train_final = np.hstack([
        X_train_tfidf_combined.toarray(),   # Konvertiere die TF-IDF-Features in ein Array
        train_w2v_emb,                      # Füge die Word2Vec-Embeddings hinzu
        train_crf_ohe,                      # Füge die CRF-Labels hinzu
        train_word_counts,                  # Füge die Wortanzahlen hinzu
        train_empty_feat                    # Füge das leere Text-Flag hinzu
    ])
    X_test_final = np.hstack([
        X_test_tfidf_combined.toarray(),
        test_w2v_emb,
        test_crf_ohe,
        test_word_counts,
        test_empty_feat
    ])

    # Extrahieren der Trainings- und Testlabels
    y_train = train_df['TAR'].values  # Trainingslabels
    y_test = test_df['TAR'].values  # Testlabels

    # Trainieren des Random Forest Klassifikators mit Grid Search
    rf = RandomForestClassifier(class_weight='balanced', random_state=42) # Random Forest Modell initialisieren mit ausgeglichenen Klassengewichtung
    param_grid = {
        'n_estimators': [100],  # Anzahl der Bäume im Wald
        'max_depth': [None],    # Maximale Tiefe der Bäume
        'min_samples_split': [2]    # Minimale Anzahl der Samples, die erforderlich sind, um einen Knoten zu teilen
    }
    # Finden der optimalen Parameter mit Grid Search
    grid = GridSearchCV(rf, param_grid, scoring='f1_macro', cv=3) # Grid Search mit F1-Macro als Metrik und 3-facher Kreuzvalidierung
    grid.fit(X_train_final, y_train) # Führt die GridSearch auf den Trainingsdaten durch
    best_rf = grid.best_estimator_  # Speichert das beste Modell

    # Vorhersagewahrscheinlichkeiten mit dem Random Forest
    y_train_proba_rf = best_rf.predict_proba(X_train_final)    # Vorhersagewahrscheinlichkeiten für Trainingsdaten
    y_test_proba_rf = best_rf.predict_proba(X_test_final)      # Vorhersagewahrscheinlichkeiten für Testdaten

    # Setzen des Labels 'public', wenn der Text leer ist
    y_pred = best_rf.predict(X_test_final)
    y_pred = np.where(test_df['is_empty'].values, 'public', y_pred)

    # Evaluation der Vorhersagen
    labels_eval = ['group', 'individual', 'public']
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels_eval, average='micro', zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels_eval, average='macro', zero_division=0
    )
    cls_report = classification_report(y_test, y_pred, labels=labels_eval, digits=4)

    # Zusammenstellen der Ergebnisse
    results = {
        'single_weight': SINGLE_SENT_WEIGHT,
        'multi_weight': MULTI_SENT_WEIGHT,
        'ngram_size': NGRAM_SIZE,
        'pos_context_length': POS_CONTEXT_LEN,
        'top_n_similar': top_n_similar,
        'oversampling_factor': oversampling_factor,
        'micro_precision': p_micro,
        'micro_recall': r_micro,
        'micro_f1': f_micro,
        'macro_precision': p_macro,
        'macro_recall': r_macro,
        'macro_f1': f_macro,
        'cls_report': cls_report,
        'crf_report': crf_report,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_test_proba_rf': y_test_proba_rf,  # Hinzufügen der Vorhersagewahrscheinlichkeiten
        'test_sentence_counts': test_sentence_counts,
        'rf_classes': best_rf.classes_
    }
    return results


############################################################
# Hauptprogramm
############################################################

if __name__ == "__main__":
    # Einlesen der Daten
    df = read_data(DATA_PATH)
    df['sentence_count'] = df['description'].apply(sentence_count)

    # Aufteilen der Daten in Trainings- und Testset
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['TAR']
    )

    # Definieren von Schwellenwerten für die Nachbearbeitung
    thresholds = np.linspace(0.5, 0.9, 9)  # Werte von 0.5 bis 0.9 in Schritten von 0.05
    all_results_with_threshold = []  # Liste zur Speicherung aller Ergebnisse mit Thresholds
    base_results = []  # Liste zur Speicherung der Basis-Ergebnisse

    # Variablen zur Speicherung des besten Modells
    best_model_info = None
    best_macro_f1 = -np.inf

    # Schleife über alle Kombinationen der Hyperparameter
    for s_w in SINGLE_SENT_WEIGHTS:
        for m_w in MULTI_SENT_WEIGHTS:
            for ngram in CRF_NGRAM_SIZES:
                for pos_len in POS_CONTEXT_LENGTHS:
                    for top_n_s in TOPN_SIMILAR_VALUES:
                        for ov_factor in OVERSAMPLING_FACTORS:
                            # Führen Sie das Experiment mit der aktuellen Parameterkombination durch
                            res = run_experiment(
                                s_w, m_w, ngram, pos_len,
                                top_n_s, ov_factor,
                                train_df.copy(), test_df.copy(),
                                CRF_LABELS
                            )
                            base_results.append(res)

                            y_test = res['y_test']
                            y_test_proba_rf = res['y_test_proba_rf']
                            test_sentence_counts = res['test_sentence_counts']
                            rf_classes = res['rf_classes']
                            labels_eval = ['group', 'individual', 'public']
                            pub_idx = np.where(rf_classes == 'public')[0][0]

                            # Schleife über alle definierten Schwellenwerte
                            for th in thresholds:
                                y_adjusted = []
                                for i, (probs, count) in enumerate(zip(y_test_proba_rf, test_sentence_counts)):
                                    pub_proba = probs[pub_idx]
                                    if count > 1 and pub_proba < th:
                                        # Sortiere die Labels nach Wahrscheinlichkeit absteigend
                                        sorted_labels = sorted(zip(rf_classes, probs), key=lambda x: x[1], reverse=True)
                                        best_label, best_val = sorted_labels[0]
                                        second_label, second_val = sorted_labels[1]

                                        if best_label == 'public':
                                            if second_label in ['group', 'individual']:
                                                y_adjusted.append(second_label)
                                            else:
                                                y_adjusted.append(best_label)
                                        else:
                                            y_adjusted.append(best_label)
                                    else:
                                        pred_label = rf_classes[np.argmax(probs)]
                                        y_adjusted.append(pred_label)

                                # Berechne die Evaluationsmetriken für die angepassten Vorhersagen
                                p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
                                    y_test, y_adjusted, labels=labels_eval, average='micro', zero_division=0
                                )
                                p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
                                    y_test, y_adjusted, labels=labels_eval, average='macro', zero_division=0
                                )

                                # Speichern der Ergebnisse mit dem aktuellen Schwellenwert
                                result = {
                                    'single_weight': s_w,
                                    'multi_weight': m_w,
                                    'ngram_size': ngram,
                                    'pos_context_length': pos_len,
                                    'top_n_similar': top_n_s,
                                    'oversampling_factor': ov_factor,
                                    'threshold': th,
                                    'micro_precision': p_micro,
                                    'micro_recall': r_micro,
                                    'micro_f1': f_micro,
                                    'macro_precision': p_macro,
                                    'macro_recall': r_macro,
                                    'macro_f1': f_macro
                                }
                                all_results_with_threshold.append(result)

                                # Überprüfen, ob dies das beste bisher gefundene Modell ist
                                if f_macro > best_macro_f1:
                                    best_macro_f1 = f_macro
                                    best_model_info = {
                                        'parameters': result,
                                        'y_test': y_test,
                                        'y_pred': y_adjusted,
                                        'rf_classes': rf_classes
                                    }

    # Erstellen eines DataFrames aus den Ergebnissen
    df_results_threshold = pd.DataFrame(all_results_with_threshold)

    if best_model_info is not None:
        # Berechnen des Klassifikationsberichts für das beste Modell
        report = classification_report(
            best_model_info['y_test'],
            best_model_info['y_pred'],
            labels=labels_eval,
            target_names=labels_eval,
            zero_division=0,
            output_dict=True
        )

        # Konvertieren des Berichts in einen DataFrame
        df_report = pd.DataFrame(report).transpose()

        # Optional: Runden der numerischen Werte für bessere Lesbarkeit
        df_report = df_report.round(4)

        # Speichern der Ergebnisse in der Ausgabedatei
        original_stdout = sys.stdout
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            sys.stdout = f
            print("Ergebnisse mit verschiedenen Thresholds:")
            print(df_results_threshold.to_string(index=False))
            print("\nBeste gefundene Kombination mit Threshold:")
            print(best_model_info['parameters'])
            print("\nKlassifikationsbericht des besten Modells:")
            print(df_report.to_string())
        sys.stdout = original_stdout

        print("Ergebnisse wurden in", OUTPUT_FILE, "gespeichert.")
    else:
        print("Kein Modell gefunden.")
