import operator

N=10
documents=[]
for i in range(10):
    searchfile = open("Inverted Index\T"+str(i+1)+".txt", "r")
    doci=[]
    for line in searchfile:
        line=line.split(":")
        doci.append(line[0])
    documents.append(doci)
    searchfile.close()
print(documents)

Nw=dict()
docs=[]
for i in range(10):
    docs=docs+documents[i]

docs=list(set(docs))

def compute(Nw):
    return (N-Nw+0.5)/(Nw+0.5)

table=dict()
for x in docs:
    nw=0
    list1=[]
    for i in documents:
        if x in i:
            nw+=1
    list1.append(nw)
    calc=compute(nw)
    list1.append(calc)
    table[x]=list1
print(table)

#query=input("Enter query :").split(" ")

pd=dict()


# ------------------------------------------------------------------------------------------------------------

"""

# from nltk import FreqDist


files = [r'Inverted Index/T1.txt',
         r'Inverted Index/T2.txt',
         r'Inverted Index/T3.txt',
         r'Inverted Index/T4.txt',
         r'Inverted Index/T5.txt',
         r'Inverted Index/T6.txt',
         r'Inverted Index/T7.txt',
         r'Inverted Index/T8.txt',
         r'Inverted Index/T9.txt',
         r'Inverted Index/T10.txt']

xfiles = [(i[len(i) - i[::-1].index('/'):]) for i in files]



# Extract text from file, return text
def extract_text(fname):
    myf = open(fname, "rb")
    text = myf.read().decode(errors='replace')
    return text


# doing analysis

def uniqueWordRatio(token):
    return str(len(set(token)) / len(token))


# Checking distribution of words

def freqDist(tokens, title):
    fdist1 = FreqDist(tokens)
    fdist1.plot(50, cumulative=True, title=title)
"""


