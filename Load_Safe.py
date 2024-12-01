import pandas as pd
import csv


# Lädt die Daten als DataFrame
def load_data(file_path):
    """
    Lädt die CSV-Datei und führt grundlegende Bereinigungen durch.

    :param file_path: Pfad zur CSV-Datei
    :return: Bereinigter DataFrame
    """
    try:
        # Lade die Datei mit angegebenen Parametern
        df = pd.read_csv(
            file_path,
            delimiter=";",
            quotechar='"',
            dtype={'id': str, "description": str, 'TAR': str},
            skip_blank_lines=False,
            on_bad_lines='warn'
        )
    except Exception as e:
        raise ValueError(f"Fehler beim Laden der Datei: {e}")

    # Überprüfen, ob die erforderlichen Spalten vorhanden sind
    required_columns_original = ['id', 'description', 'TAR']
    missing_columns = [col for col in required_columns_original if col not in df.columns]
    if missing_columns:
        print(f"Fehlende Spalten: {missing_columns}")
        print("Vorschau der gefundenen Spalten:", df.columns.tolist())
        raise ValueError(f"Die CSV-Datei muss die Spalten {required_columns_original} enthalten.")

    # Spalte 'description' in 'text' umbenennen
    df.rename(columns={'description': 'text'}, inplace=True)

    # Entferne Zeilenumbrüche innerhalb von Textspalten und ersetze sie durch Leerzeichen
    required_columns = ['id', 'text', 'TAR']
    for col in required_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(r'\r?\n', ' ', regex=True)

    # Entferne komplett leere Zeilen
    df.dropna(how="all", inplace=True)

    # Überprüfe auf problematische Zeilen mit fehlenden Werten
    invalid_rows = df[df[required_columns].isnull().any(axis=1)]
    if not invalid_rows.empty:
        print("Fehlerhafte Zeilen mit fehlenden Werten in den erforderlichen Spalten:")
        print(invalid_rows)
        raise ValueError("Die CSV-Datei enthält fehlerhafte Zeilen mit fehlenden Werten. Bitte prüfen.")

    return df


# Lädt die vorverarbeiteten Daten wieder als DataFrame
def load_processed_data(file_path):
    """
    Lädt die vorverarbeitete CSV-Datei ein und stellt sicher, dass die Spalten 'tokens' und 'pos_tags'
    korrekt als Listen eingelesen werden.

    :param file_path: Pfad zur vorverarbeiteten CSV-Datei
    :return: DataFrame mit korrekt eingelesenen Daten
    """

    # Lade die CSV-Datei
    df = pd.read_csv(
        file_path,
        delimiter=';',
        quotechar='"',
        dtype={'id': str, 'text': str, 'TAR': str, 'predicted_label': str},
        converters={
            'tokens': lambda x: x.split(',') if pd.notnull(x) else [],
            'pos_tags': lambda x: x.split(',') if pd.notnull(x) else []
        }
    )
    return df

# Speichert die Daten, welche zuvor gespeichert wurden
def save_processed_data(df, file_path):
    """
    Speichert die verarbeiteten Daten in einer CSV-Datei.
    Konvertiert komplexe Datentypen wie Listen in Strings, um sie korrekt speichern zu können.

    :param df: DataFrame mit den zu speichernden Daten
    :param file_path: Zielpfad der CSV-Datei
    """
    try:
        # Konvertiere Listen in Strings, um sie korrekt in die CSV zu schreiben
        df['tokens'] = df['tokens'].apply(lambda x: str(x) if isinstance(x, list) else x)
        df['pos_tags'] = df['pos_tags'].apply(lambda x: str(x) if isinstance(x, list) else x)

        # Speichere das DataFrame
        df.to_csv(file_path, index=False, sep=';', quotechar='"', quoting=csv.QUOTE_ALL)
        print(f"Daten erfolgreich in {file_path} gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern der Datei {file_path}: {e}")