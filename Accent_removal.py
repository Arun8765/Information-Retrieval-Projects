from removeaccents import removeaccents
filer = open(r"Document Preprocessing\Text3.txt" , 'r')
filew = open(r"Preprocessed\Text3_preprocessed.txt" , "w")
myline = filer.readline()
while myline:
    myline = removeaccents.remove_accents(myline)
    filew.write(myline)
    myline=filer.readline()

filer.close()
filew.close()



