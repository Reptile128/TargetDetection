import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Lade die CSV-Datei
def load_data(file_path):
    # Lade CSV mit dem Trennzeichen ';'
    data = pd.read_csv(file_path, delimiter=';')
    return data

# Trainiere ein Neuronales Netzwerk-Modell
def train_nn(data):

    # Behandle fehlende Werte in der 'String'-Spalte
    data['text'] = data['text'].fillna("")  # Ersetze NaN durch leere Zeichenkette

    # Extrahiere Merkmale (Text-Spalte) und Labels (Label-Spalte)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['text'])  # Transformiere den Text in TF-IDF Merkmalsvektoren
    y = data['TAR']

    # Teile die Daten in Trainings- und Testsets auf
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialisiere das Neuronale Netzwerk-Modell
    nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

    # Trainiere das Neuronale Netzwerk-Modell
    nn_model.fit(X_train, y_train)

    # Mache Vorhersagen für das Testset
    y_pred = nn_model.predict(X_test)

    # Bewerte das Modell
    print("Genauigkeit:", accuracy_score(y_test, y_pred))
    print("Klassifikationsbericht:\n", classification_report(y_test, y_pred))

    return nn_model

if __name__ == "__main__":
    # Pfad zur CSV-Datei
    file_path = "Data/Tweets_Complete_Preprocessing.csv"  # Ersetze mit deinem Dateipfad

    # Lade die Daten
    data = load_data(file_path)

    # Trainiere das Neuronale Netzwerk-Modell
    nn_model = train_nn(data)
