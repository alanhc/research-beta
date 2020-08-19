import os

def getBaseName(path):
    filename = os.path.basename(path)
    filename = filename.split('.')
    return filename

def createFolder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
    