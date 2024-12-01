import csv #ermoeglicht das einfache arbeiten mit csv dateien.
from logging import exception

speicher = []
def lesen (source):
    speicher.clear()
    with open(source, mode="r", encoding="utf-8") as csvfile:             #Oeffnet die datei in die Variable csvfile
        datei = csv.reader(csvfile, delimiter=';')      #Der csv.reader erstellt eine liste. die trennuung der eintraege inerhalb der csv datei muss beim delimiter angegeben werden
        for row in datei:                               #jede einzelne reihe wird nun aus der datei gelesen, und dem speicher einzeln eingefuegt
            speicher.append(row)
    csvfile.close()



    return speicher


