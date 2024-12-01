import LinkDetection
import Read
import Uebersetzung

liste = Read.lesen("C:\\Users\\tomli\Downloads\\tar.csv")

for i in range(1000):
    print(liste[i][1])

print("\t")
liste = LinkDetection.LinkEntfernung(liste)

for i in range(1000):
    print(liste[i][1])

#sprache = Uebersetzung.spracherkennung(liste)

#liste = Uebersetzung.ueberstzung(liste, sprache)

#print("\t")

#for i in range(len(liste)):
  #print(liste[i])