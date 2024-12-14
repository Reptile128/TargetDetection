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
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.stem.snowball import GermanStemmer

############################################################
# Parameter-Definitionen
############################################################

DATA_PATH = 'Data/D_verarbeitet_oa.csv'  # Pfad zur bereinigten Datendatei
FAMILIENNAMEN_PATH = 'Data/Familiennamen.txt'  # Pfad zur Datei mit Familiennamen
VORNAMEN_PATH = 'Data/Vornamen.txt'  # Pfad zur Datei mit Vornamen

# Arrays zur Erprobung verschiedener Hyperparameter-Kombinationen
TOPN_SIMILAR_VALUES = [10]  # Verschiedene Werte für TOPN_SIMILAR
SINGLE_SENT_WEIGHTS = [1.0]  # Gewichtungen für Ein-Satz-Tweets
MULTI_SENT_WEIGHTS = [0.4]  # Gewichtungen für Mehr-Satz-Tweets
CRF_NGRAM_SIZES = [10]  # n-Gram-Größen für CRF-Features              #10
POS_CONTEXT_LENGTHS = [2]  # Kontextlängen für POS-Features
OVERSAMPLING_FACTORS = [5]  # Oversampling-Faktoren für CRF         #5
CRF_LABELS = ['group', 'individual', 'public']  # Mögliche CRF-Labels für die Vorhersage von den Gruppen
CHECK_NOUN_ART_PRON_ENT = True  # Flag zur Filterung von Tokens
OUTPUT_FILE = 'Data/ergebnisse_oa.txt'  # Pfad zur Ausgabedatei

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
    df.rename(columns={'translated_description': 'description'}, inplace=True)
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
    tfidf = TfidfVectorizer()           # Initialisiert TF-IDF-Vektoriser, um TF-IDF zu erstellen
    X = tfidf.fit_transform(texts)      # Passt de Vektorisier an die Eingabetexte an
    row_weights = np.array([single_weight if c == 1 else multi_weight for c in sentence_counts])    # Weißt den Daten eine Gewichtung hinzu, je nach Multi/Einzelsatz
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
    X = tfidf.transform(texts)                  # transformiert die neuen Texte in die TF-IDF-Matrix
    row_weights = np.array([single_weight if c == 1 else multi_weight for c in sentence_counts])    # Weißt den Daten eine Gewichtung hinzu, je nach Multi/Einzelsatz
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

    # Verbinde die Wörter mit ihren IDF-Werte und sortiere sie nach Relevanz (IDF-Wert)
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
        str: Die Differenz zwischen Stamm und Orginalwort.
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
    Extrahiert Nominalphrasen (NP) aus dem SpaCy-Dokument, inkl. Pronomen und Artikeln.

    Args:
        doc (spacy.tokens.Doc): Das SpaCy-Dokument.

    Returns:
        list: Liste von NPs, wobei jede NP eine Liste aus den Tupeln.
    """
    nps = []
    # Extrahiere NPs aus den Noun-Chunks von SpaCy
    for chunk in doc.noun_chunks:
        np_tokens = []
        for t in chunk:
            np_tokens.append((t.text.lower(), t.pos_, t.ent_type_ if t.ent_type_ else 'O'))
        nps.append(np_tokens)

    # Füge Pronomen und Artikel hinzu, die nicht bereits in den Noun-Chunks enthalten sind
    existing_np_words = set(w for np_chunk in nps for w, p, e in np_chunk)
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
    nps = extract_np_spans(doc)
    np_words = set(ww for np_chunk in nps for ww, pp, ee in np_chunk)

    filtered = []
    for (w, p, isn, ent) in doc_tokens:
        if w == '.':
            break  # Stoppt die Filterung bei einem Punkt
        in_np = w in np_words
        if in_np:
            filtered.append((w, p, isn, ent))  # NPs immer hinzufügen
        else:
            if CHECK_NOUN_ART_PRON_ENT:
                # Hinzufügen, wenn POS in [NOUN, DET, PRON] oder Entität vorhanden
                if (p in ['NOUN', 'DET', 'PRON'] or ent != 'O'):
                    filtered.append((w, p, isn, ent))
    return filtered


def ngram_tokens_overlapping(doc_tokens, n):
    """
    Erzeugt n-Gramme von Tokens mit Überlappung.

    Args:
        doc_tokens (list): Liste von Token-Tupeln (w, p, isn, ent).
        n (int): Größe des n-Gramms.

    Returns:
        list: Liste von n-Gram-Token-Tupeln.
    """
    merged = []
    for i in range(len(doc_tokens) - n + 1):
        chunk = doc_tokens[i:i + n]
        w_merged = "_".join([x[0] for x in chunk])  # Zusammengesetztes Wort
        p_first = chunk[0][1]  # POS des ersten Tokens
        is_name_any = any(x[2] for x in chunk)  # Prüft, ob eines der Tokens ein Name ist
        ent_first = chunk[0][3]  # Entität des ersten Tokens
        merged.append((w_merged, p_first, is_name_any, ent_first))
    return merged


def ner_boost_pos_text(doc_tokens):
    """
    Erstellt einen Text mit wiederholten Wörtern, um NER-Informationen zu verstärken.

    Args:
        doc_tokens (list): Liste von Token-Tupeln (w, p, isn, ent).

    Returns:
        str: Verstärkter Text basierend auf POS-Tags und Entitäten.
    """
    words = []
    for w, p, isn, ent in doc_tokens:
        if p in ['NOUN', 'DET', 'PRON']:
            if ent != 'O':
                words.extend([w, w])  # Wiederhole Wörter mit Entitäten
            else:
                words.append(w)
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
    Wandelt ein Label in ein One-Hot-Encoded-Format um.

    Args:
        lab (str): Das zu kodierende Label.
        crf_labels (list): Liste der möglichen Labels.

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
        tuple: Neue überabgestimmte Listen von CRF-Features und Labels.
    """
    single_rep = max(1, int(round(single_weight * oversampling_factor)))
    multi_rep = max(1, int(round(multi_weight * oversampling_factor)))

    X_new = []
    Y_new = []
    for x_seq, y_seq, scount in zip(X_crf, Y_crf, sentence_counts):
        if scount == 1:
            for _ in range(single_rep):
                X_new.append(x_seq)
                Y_new.append(y_seq)
        else:
            for _ in range(multi_rep):
                X_new.append(x_seq)
                Y_new.append(y_seq)
    return X_new, Y_new


def get_context_features(w_index, sent_toks, pos_context_length):
    """
    Extrahiert Kontext-Features basierend auf POS-Tags und Entitäten.

    Args:
        w_index (int): Index des aktuellen Wortes in der Satzliste.
        sent_toks (list): Liste von Token-Tupeln im Satz.
        pos_context_length (int): Länge des POS-Kontexts.

    Returns:
        dict: Kontext-Features.
    """
    features = {}
    for offset in range(1, pos_context_length + 1):
        # Kontext vor dem aktuellen Wort
        if w_index - offset >= 0:
            pw, pp, pe = sent_toks[w_index - offset]
            features[f'-{offset}:pos'] = pp
            features[f'-{offset}:ent'] = str(pe != 'O')
        # Kontext nach dem aktuellen Wort
        if w_index + offset < len(sent_toks):
            nw, np_, ne = sent_toks[w_index + offset]
            features[f'+{offset}:pos'] = np_
            features[f'+{offset}:ent'] = str(ne != 'O')
    return features


def extract_crf_features_for_doc(doc_tokens, doc, model_w2v, idf_all, feature_names_all, pos_context_length, vornamen,
                                 nachnamen, top_n_similar):
    """
    Extrahiert Features für das CRF-Modell aus einem Dokument.

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
    nps = extract_np_spans(doc)
    token2np = {}
    for np_chunk in nps:
        for ww, pp, ee in np_chunk:
            token2np[ww] = np_chunk

    sentences = list(doc.sents)
    sents_tokens = []
    doc_linear_tokens = []
    for sent in sentences:
        stoks = []
        for tt in sent:
            w_ = tt.text.lower()
            p_ = tt.pos_
            e_ = tt.ent_type_ if tt.ent_type_ else 'O'
            stoks.append((w_, p_, e_))
            doc_linear_tokens.append((w_, p_, e_))
        sents_tokens.append(stoks)

    matched_indices = []
    used = [False] * len(doc_linear_tokens)
    for idx, (w, p, isn, ent) in enumerate(doc_tokens):
        for j, (ww, wp, we) in enumerate(doc_linear_tokens):
            if not used[j] and ww == w:
                sent_id = 0
                c = 0
                for si, st in enumerate(sents_tokens):
                    if j < c + len(st):
                        sent_id = si
                        w_id = j - c
                        break
                    c += len(st)
                matched_indices.append((idx, sent_id, w_id))
                used[j] = True
                break

    features_list = []
    for idx, (w, p, isn, ent) in enumerate(doc_tokens):
        token_features = {
            'bias': 1.0,  # Bias-Feature
            'word.lower()': w,  # Kleinbuchstaben des Wortes
            'pos': p,  # POS-Tag
            'is_name': str(isn),  # Flag, ob es ein Name ist
            'ent_type': ent  # Entitätstyp
        }

        # NP-Features
        if w in token2np:
            np_tokens = token2np[w]
            # Extrahiere Artikel
            det_words = [x[0] for x in np_tokens if x[1] == 'DET']
            if det_words:
                token_features['np_det'] = '_'.join(det_words)

            # Extrahiere und stemme Adjektive
            adj_tokens = [x[0] for x in np_tokens if x[1] == 'ADJ']
            if adj_tokens:
                stemmed_adjs = [stem_adj(a) for a in adj_tokens]
                token_features['np_adjs_stemmed'] = '_'.join(stemmed_adjs)

            # Extrahiere und stemme Nomen
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

        # Word2Vec-Features
        if model_w2v is not None and idf_all is not None and feature_names_all is not None:
            w2v_feats = get_word2vec_features_nouns(model_w2v, w, idf_all, feature_names_all, p, top_n_similar)
            token_features.update(w2v_feats)

        # Kontext-Features
        sent_id = None
        w_id = None
        for m in matched_indices:
            if m[0] == idx:
                sent_id = m[1]
                w_id = m[2]
                break

        if sent_id is not None and w_id is not None:
            sent_toks = sents_tokens[sent_id]
            c_feats = get_context_features(w_id, sent_toks, pos_context_length)
            token_features.update(c_feats)

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
    train_df['doc_tuple'] = train_df['description'].apply(lambda x: process_text(x, nlp, vornamen, nachnamen))
    train_df['doc'] = train_df['doc_tuple'].apply(lambda x: x[0])
    train_df['doc_tokens'] = train_df['doc_tuple'].apply(lambda x: x[1])
    train_df.drop(columns=['doc_tuple'], inplace=True)

    test_df['doc_tuple'] = test_df['description'].apply(lambda x: process_text(x, nlp, vornamen, nachnamen))
    test_df['doc'] = test_df['doc_tuple'].apply(lambda x: x[0])
    test_df['doc_tokens'] = test_df['doc_tuple'].apply(lambda x: x[1])
    test_df.drop(columns=['doc_tuple'], inplace=True)

    # Filtern der Tokens für das CRF-Modell
    train_df['doc_tokens_crf'] = train_df.apply(lambda row: filter_tokens_for_crf(row['doc_tokens'], row['doc']),
                                                axis=1)
    test_df['doc_tokens_crf'] = test_df.apply(lambda row: filter_tokens_for_crf(row['doc_tokens'], row['doc']), axis=1)

    # Erzeugen von n-Grammen
    train_df['doc_tokens_crf_ng'] = train_df['doc_tokens_crf'].apply(
        lambda dt: ngram_tokens_overlapping(dt, NGRAM_SIZE))
    test_df['doc_tokens_crf_ng'] = test_df['doc_tokens_crf'].apply(lambda dt: ngram_tokens_overlapping(dt, NGRAM_SIZE))

    # Trainieren des Word2Vec-Modells
    sentences = [[w for w, p, isn, ent in doc] for doc in train_df['doc_tokens_crf_ng']]
    model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=20)

    # Erzeugen von TF-IDF-Texten
    train_texts_all = train_df['doc_tokens'].apply(join_tokens_all).tolist()
    test_texts_all = test_df['doc_tokens'].apply(join_tokens_all).tolist()

    # Erzeugen von POS-gestärkten Texten
    train_texts_pos = train_df['doc_tokens'].apply(ner_boost_pos_text).tolist()
    test_texts_pos = test_df['doc_tokens'].apply(ner_boost_pos_text).tolist()

    # Extrahieren der Satzanzahlen
    train_sentence_counts = train_df['sentence_count'].values
    test_sentence_counts = test_df['sentence_count'].values

    # Vektorisieren und Gewichtung der TF-IDF-Features
    X_train_all, tfidf_all = vectorize_and_weight(train_texts_all, train_sentence_counts, SINGLE_SENT_WEIGHT,
                                                  MULTI_SENT_WEIGHT)
    X_test_all = transform_and_weight(tfidf_all, test_texts_all, test_sentence_counts, SINGLE_SENT_WEIGHT,
                                      MULTI_SENT_WEIGHT)

    X_train_pos, tfidf_pos = vectorize_and_weight(train_texts_pos, train_sentence_counts, SINGLE_SENT_WEIGHT,
                                                  MULTI_SENT_WEIGHT)
    X_test_pos = transform_and_weight(tfidf_pos, test_texts_pos, test_sentence_counts, SINGLE_SENT_WEIGHT,
                                      MULTI_SENT_WEIGHT)

    feature_names_all = np.array(tfidf_all.get_feature_names_out())  # Alle TF-IDF-Feature-Namen
    idf_all = tfidf_all.idf_  # IDF-Werte

    # Extrahieren von CRF-Features und Labels
    X_train_crf = []
    y_train_crf = []
    for i, row in train_df.iterrows():
        feats = extract_crf_features_for_doc(
            row['doc_tokens_crf_ng'], row['doc'], model_w2v, idf_all,
            feature_names_all, POS_CONTEXT_LEN, vornamen, nachnamen, top_n_similar
        )
        labels = [row['TAR']] * len(row['doc_tokens_crf_ng'])  # Label für jedes Token
        X_train_crf.append(feats)
        y_train_crf.append(labels)

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
        algorithm='lbfgs', c1=0.1, c2=0.1,
        max_iterations=100, all_possible_transitions=True
    )
    crf.fit(X_train_crf_ov, y_train_crf_ov)

    # Vorhersagen mit dem CRF-Modell
    y_pred_crf = crf.predict(X_test_crf)
    y_pred_crf_tweet = [tokens[0] if len(tokens) > 0 else 'public' for tokens in y_pred_crf]
    crf_report = crf_metrics.flat_classification_report(y_test_crf, y_pred_crf, digits=4)

    # Erzeugen von Word2Vec-Embeddings für die Dokumente
    train_w2v_emb = np.array([
        np.mean(
            [model_w2v.wv[w] for w, p, i, e in doc if w in model_w2v.wv] or [np.zeros(model_w2v.wv.vector_size)],
            axis=0
        )
        for doc in train_df['doc_tokens_crf_ng']
    ])
    test_w2v_emb = np.array([
        np.mean(
            [model_w2v.wv[w] for w, p, i, e in doc if w in model_w2v.wv] or [np.zeros(model_w2v.wv.vector_size)],
            axis=0
        )
        for doc in test_df['doc_tokens_crf_ng']
    ])

    # One-Hot-Encoding der CRF-Labels
    y_train_crf_tweet = [t[0] if len(t) > 0 else 'public' for t in y_train_crf]

    def ohe(l):
        return one_hot_label(l, crf_labels)

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
        X_train_tfidf_combined.toarray(),
        train_w2v_emb,
        train_crf_ohe,
        train_word_counts,
        train_empty_feat
    ])
    X_test_final = np.hstack([
        X_test_tfidf_combined.toarray(),
        test_w2v_emb,
        test_crf_ohe,
        test_word_counts,
        test_empty_feat
    ])

    y_train = train_df['TAR'].values  # Trainingslabels
    y_test = test_df['TAR'].values  # Testlabels

    # Trainieren des Random Forest Klassifikators mit Grid Search
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    param_grid = {
        'n_estimators': [100],
        'max_depth': [None],
        'min_samples_split': [2]
    }
    grid = GridSearchCV(rf, param_grid, scoring='f1_macro', cv=3)
    grid.fit(X_train_final, y_train)
    best_rf = grid.best_estimator_

    # Vorhersagewahrscheinlichkeiten mit dem Random Forest
    y_train_proba = best_rf.predict_proba(X_train_final)
    y_test_proba = best_rf.predict_proba(X_test_final)

    def build_second_stage_features(proba, counts):
        """
        Baut Features für die zweite Modellstufe auf.

        Args:
            proba (array-like): Vorhersagewahrscheinlichkeiten.
            counts (array-like): Satzanzahlen.

        Returns:
            array-like: Kombinierte Features.
        """
        return np.hstack([proba, counts.reshape(-1, 1)])

    # Aufbau der Features für das zweite Modell (Logistic Regression)
    X_train_second = build_second_stage_features(y_train_proba, train_sentence_counts)
    X_test_second = build_second_stage_features(y_test_proba, test_sentence_counts)

    # Trainieren des Logistic Regression Modells als zweite Stufe
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train_second, y_train)
    y_test_proba_lr = lr.predict_proba(X_test_second)
    y_final = lr.predict(X_test_second)

    # Setzen des Labels 'public', wenn der Text leer ist
    y_final = np.where(test_df['is_empty'].values, 'public', y_final)

    # Evaluation der Vorhersagen
    labels_eval = ['group', 'individual', 'public']
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_test, y_final, labels=labels_eval, average='micro'
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_test, y_final, labels=labels_eval, average='macro'
    )
    cls_report = classification_report(y_test, y_final, labels=labels_eval, digits=4)

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
        'y_test_proba_lr': y_test_proba_lr,
        'test_sentence_counts': test_sentence_counts,
        'lr_classes': lr.classes_
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
                            y_test_proba_lr = res['y_test_proba_lr']
                            test_sentence_counts = res['test_sentence_counts']
                            lr_classes = res['lr_classes']
                            labels_eval = ['group', 'individual', 'public']
                            pub_idx = np.where(lr_classes == 'public')[0][0]

                            # Schleife über alle definierten Schwellenwerte
                            for th in thresholds:
                                y_adjusted = []
                                for i, (probs, count) in enumerate(zip(y_test_proba_lr, test_sentence_counts)):
                                    pub_proba = probs[pub_idx]
                                    if count > 1 and pub_proba < th:
                                        # Sortiere die Labels nach Wahrscheinlichkeit absteigend
                                        sorted_labels = sorted(zip(lr_classes, probs), key=lambda x: x[1], reverse=True)
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
                                        pred_label = lr_classes[np.argmax(probs)]
                                        y_adjusted.append(pred_label)

                                # Berechne die Evaluationsmetriken für die angepassten Vorhersagen
                                p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
                                    y_test, y_adjusted, labels=labels_eval, average='micro'
                                )
                                p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
                                    y_test, y_adjusted, labels=labels_eval, average='macro'
                                )

                                # Speichern der Ergebnisse mit dem aktuellen Schwellenwert
                                all_results_with_threshold.append({
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
                                })

    # Erstellen eines DataFrames aus den Ergebnissen
    df_results_threshold = pd.DataFrame(all_results_with_threshold)

    # Finden der besten Kombination basierend auf dem Macro F1-Score
    best_with_threshold = df_results_threshold.loc[df_results_threshold['macro_f1'].idxmax()]

    # Speichern der Ergebnisse in der Ausgabedatei
    original_stdout = sys.stdout
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print("Ergebnisse mit verschiedenen Thresholds:")
        print(df_results_threshold.to_string(index=False))
        print("\nBeste gefundene Kombination mit Threshold:")
        print(best_with_threshold)
    sys.stdout = original_stdout

    print("Ergebnisse wurden in", OUTPUT_FILE, "gespeichert.")
