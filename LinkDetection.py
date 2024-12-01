import re

import Read


def LinkEntfernung(liste):
    text = ""
    neue_liste = []
    for i in range(len(liste)):                                         #schleife um alle eintraege einzel zu bearbeiten
        for j in range(len(liste[i])):
            text = str(liste[i][j])

            if "http" in text or "https" in text or "www." in text:     #vorauswahl um unnoetige bearbeitungen zu sparen

                RA_Muster = r"(https?://|www\.)\S+[^\]']"               #Hier wird das musster fuer regulaere ausdruecke festgelegt

                neuer_text = re.sub(RA_Muster, '', text)           #Der link wird nun gezielt durch nichts ersetzt. Leerzeichen vor und hinter dem link bleiben aber da, sollte keine Probleme machen, wollte es aber nur mal anmerken

                liste[i][j] = neuer_text                                #Bearbeiteter Text wird nun wieder an der gleichen stelle eingefuegt, sodass die Form der Liste erhalten bleibt

    return liste
