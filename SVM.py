import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Lade die CSV-Datei
def load_data(file_path):
    # Lade CSV mit dem Trennzeichen ';'
    data = pd.read_csv(file_path, delimiter=';')
    return data

# Trainiere ein SVM-Modell
def train_svm(data):

    # Behandle fehlende Werte in der 'String'-Spalte
    data['text'] = data['text'].fillna("")  # Ersetze NaN durch leere Zeichenkette

    # Extrahiere Merkmale (Text-Spalte) und Labels (Label-Spalte)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['text'])  # Transformiere den Text in TF-IDF Merkmalsvektoren
    y = data['TAR']

    # Teile die Daten in Trainings- und Testsets auf
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialisiere das SVM-Modell
    svm_model = SVC(kernel='linear', random_state=42)

    # Trainiere das SVM-Modell
    svm_model.fit(X_train, y_train)

    # Mache Vorhersagen für das Testset
    y_pred = svm_model.predict(X_test)

    # Bewerte das Modell
    print("Genauigkeit:", accuracy_score(y_test, y_pred))
    print("Klassifikationsbericht:\n", classification_report(y_test, y_pred))

    return svm_model

if __name__ == "__main__":
    # Pfad zur CSV-Datei
    file_path = "Data/Tweets_edited_wo_translation_correction_stopwordremoval.csv"  # Ersetze mit deinem Dateipfad

    # Lade die Daten
    data = load_data(file_path)

    # Trainiere das SVM-Modell
    svm_model = train_svm(data)
