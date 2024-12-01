import LinkDetection
import Read

liste = Read.lesen("")

for i in range(1000):
    print(liste[i][1])

print("\t")
liste = LinkDetection.LinkEntfernung(liste)

for i in range(1000):
    print(liste[i][1])
