from langdetect import detect
from googletrans import Translator
from deep_translator import GoogleTranslator


erkannte_sprache = []

def spracherkennung(liste):

    for i in range(len(liste)):

        for j in range(len(liste[i])):

            try:
                erkannte_sprache.append(detect(str(liste[i][j])))                                           #der Liste werden nacheinander die durch detect erkannten sprachen angefuegt. Die sprache eines Listeneintrages befindet sich damit an der stelle i + j

            except Exception:

               erkannte_sprache.append("de")                                                                #Die spracherkennung wirft eine Fehlermeldung wenn keine oder zu wenig worte in einem Satz vorkommen. Hiermit werden diesen ereignissen die Sprache deutsch zugeordnet


    return erkannte_sprache


def ueberstzung(liste, sprache):

    uebersetzer = GoogleTranslator()

    proxies = {"https": "34.195.196.27:8080",
              "http": "34.195.196.27:8080"}

    for i in range(len(liste)):
        for j in range(len(liste[i])):
            if sprache[i + j] != "de":

                liste[i][j] = uebersetzer.translate(proxies=proxies, source=sprache[i + j], target="de", text=liste[i][j])
    return liste