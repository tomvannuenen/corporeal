#!/usr/bin/env python
import os
from os import listdir
import csv
import unicodedata
import re
import numpy as np
from numpy import linspace
import string
import sys
import stat
import hashlib 
from collections import Counter
from collections import OrderedDict
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import bigrams
from nltk import trigrams
from sklearn.feature_extraction.text import CountVectorizer
from unicodedata import category
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from gensim import corpora, models, similarities
import signal

# Use prettier matplotlib graphics
plt.style.use('ggplot')
# Removing traceback when user presses CTRL-C 
signal.signal(signal.SIGINT, lambda x,y: sys.exit(0))

def main():
    print("-------------------------------")
    print("CORPOREAL 0.6, by Tom van Nuenen")
    print("-------------------------------")   
    folder()

def folder():
    """Select input folder or jump to preset folder if user has selected one in main menu."""
    if os.path.exists("userFav.txt"):
        myDict = {}
        number = 0
        with open("userFav.txt", "r") as f:
            myDir = f.readline()
        print("Using favorited folder: " + myDir)
        main_menu(myDir)
    goOn = 0
    myDir = input("Please enter directory name containing .txt files.\nSee README for detailed instructions.\nSTR>>> ")
    while goOn == 0:
        if os.path.isdir(myDir):
            fileList, fileNo = list_textfiles(myDir)
            if not fileList:
                print("There are no .txt files in this folder. Exiting...")
                exit()        
            if fileNo == 1:
                print("Counting files...")
                print("This folder contains " + str(fileNo) + " file: " + fileList[0].split("/")[-1])
                main_menu(myDir)
            if fileNo > 1:
                print("Counting files...")
                print("This folder contains " + str(fileNo) + " files, from " + fileList[0].split("/")[-1] + " to " + fileList[-1].split("/")[-1])
                main_menu(myDir)
            goOn = 1         
        elif myDir == "x" or myDir == "exit":
            exit()
        else:
            myDir = input("That folder does not exist... Try again!\nSTR>>> ")
            
def main_menu(myDir):
    """Just the main menu."""
    userCond = 0
    userInput = input("""Please select:
    ---- PREPROCESSING ----------------------
    [0]  Removing duplicate files
    [1]  Chunking
    [2]  Stemming
    [3]  POS tagging
    [4]  POS filtering 
    [5]  Lemmatization
    ---- TEXT ANALYSIS ----------------------
    [6]  Word count
    [7]  Top words 
    [8]  Concordances
    [9]  Similar words (NOT FUNCTIONAL YET)
    [10] Top clusters (bi- or trigrams)
    [11] Distinctive words
    [12] Compare POS tags
    ---- GRAPH OUTPUT ----------------------
    [13] Custom word frequency
    [14] Lexical variety (means and TTR)
    [15] Euclidian distances
    [16] TF-IDF cosine distances
    ---- OPTIONS ---------------------------
    [u]  Add/remove user favorite 
    [x]  Exit (or CTRL-C at any time)\nINT>>> """).lower()
    while userCond == 0:
        if userInput == "0":
            userCond = 1
            duplicates(myDir)
        if userInput == "1":
            userCond = 1
            chunking(myDir)
        elif userInput == "2":
            userCond = 1
            stemmer(myDir)
        elif userInput == "3":
            userCond = 1
            tagger(myDir)
        elif userInput == "4":
            userCond = 1
            pos_filter(myDir)
        elif userInput == "5":
            userCond = 1
            lemmatizer(myDir)
        elif userInput == "6":
            userCond = 1
            word_count(myDir)
        elif userInput == "7":    
            userCond = 1
            top_words(myDir)
        elif userInput == "8":
            userCond = 1
            find_concordances(myDir)
        elif userInput == "9":
            userCond = 1
            similar_words(myDir)
        elif userInput == "10":
            userCond = 1
            find_clusters(myDir)
        elif userInput == "11":
            userCond = 1
            distinctive(myDir)
        elif userInput == "12":    
            userCond = 1
            compare_POS(myDir)
        elif userInput == "13":    
            userCond = 1
            word_find(myDir)
        elif userInput == "14":
            userCond = 1
            lexical_variety(myDir)
        elif userInput == "15":
            userCond = 1
            euclidian(myDir)
        elif userInput == "16":
            userCond = 1
            cosine(myDir)
        elif userInput == "x":
            exit()
        elif userInput == "u":            
            if not os.path.exists("userFav.txt"):
                print("Saving preference...")
                with open("userFav.txt", "a") as f:
                    f.write(myDir)
                main_menu(myDir)
            if os.path.exists("userFav.txt"):
                os.remove("userFav.txt")
                print("Favorite folder removed.")
                main_menu(myDir)
        else:
            userInput = input("Please try again or x to exit.\nINT>>> ")

# --- SUPPORTING FUNCTIONS ---

def list_textfiles(directory):
    "Return a list of filenames ending in '.txt' in DIRECTORY. Remove files that are (almost) empty."
    textFiles = []
    # We are sorting because different operating systems may list files in different orders
    for fileName in sorted(listdir(directory)):
        if fileName.endswith(".txt"):
            # 70 bytes seems a nice indicator of .txt files that have about 10 words or less
            if os.stat(directory + "/" + fileName).st_size > 70:
                textFiles.append(directory + "/" + fileName)
    fileNo = 1 + len(textFiles)
    return textFiles, fileNo

def read_file(filename):
    "Read the contents of FILENAME and return as a string."
    infile = open(filename) 
    contents = infile.read()
    infile.close()
    return contents

def split_text(filePath, n_words):    
    """Split text into chunks. Used in chunking function."""
    tokens = get_tokens(filePath)  
    chunks = []
    current_chunk_words = []
    current_chunk_word_count = 0
    for word in tokens:
        current_chunk_words.append(word)
        current_chunk_word_count += 1
        if current_chunk_word_count == n_words:
            chunks.append(' '.join(current_chunk_words))
            current_chunk_words = []
            current_chunk_word_count = 0
    chunks.append(' '.join(current_chunk_words) )        
    return chunks

def get_tokens(fn):
    """Get tokens, presented as a list, for analysis."""
    with open(fn, 'r') as f:
        text = f.read()
        lowers = text.lower()
        no_punctuation = ''.join(ch for ch in lowers if category(ch)[0] != 'P')
        # Another way of removing punct, which only works for ASCII:
        # no_punctuation = lowers.translate(string.punctuation)
        no_diacritics = ''.join(c for c in unicodedata.normalize('NFD', no_punctuation)
                  if unicodedata.category(c) != 'Mn')
        tokens = no_diacritics.split()
        # Another way of tokenizing, but seems less accurate:
        # tokens = nltk.word_tokenize(lowers)
        return tokens

def get_POS_tokens(fn):
    """Get POS-tagged tokens, presented as a list, for analysis. Removes stopwords.
    The tokens are modified based on their POS."""
    # To check which tagger we're using
    # print(nltk.tag._POS_TAGGER)   
    # For info about the tags see https://web.stanford.edu/~jurafsky/slp3/9.pdf
    with open(fn, 'r') as f:
        text = f.read()
        lowers = text.lower()
        no_punctuation = ''.join(ch for ch in lowers if category(ch)[0] != 'P')
        tokens = no_punctuation.split()
        # Results of this tagger could be improved with some extra rules.
        # Words like "get", "see" should be verbs; conjunctions like "i'm" don't
        # tag well yet either 
        pos = nltk.pos_tag(tokens)
        posTokens = []
        for tup in pos:
            posTokens.append(''.join(tup))              
    return posTokens

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def listFromAuthor(author, fileList):
    myDict = {}
    myDict[author] = []  
    for filePath in fileList:
        j = filePath.split("/")[-1].split("-")[0]
        if j == author:
            myDict[author].append(filePath)
    return myDict[author]
            

# --- MAIN FUNCTIONS ---

def duplicates(myDir):
    """Remove double files from user-defined filelist"""
    filesBySize = {}    
    textFiles, number = list_textfiles(myDir)
    for f in textFiles:
        if not os.path.isfile(f):
            continue
        size = os.stat(f).st_size
        if size < 100:
            continue
        if size in filesBySize:
            a = filesBySize[size]
        else:
            a = []
            filesBySize[size] = a
        a.append(f)
    print('Scanning directory "%s"....' % myDir)
#    print('Finding potential dupes...')
    potentialDupes = []
    potentialCount = 0
    trueType = type(True)
    sizes = list(filesBySize.keys())
    sizes.sort()
    for k in sizes:
        inFiles = filesBySize[k]
        outFiles = []
        hashes = {}
        if len(inFiles) is 1: continue
        #print('Testing %d files of size %d...' % (len(inFiles), k))
        for fileName in inFiles:
            if not os.path.isfile(fileName):
                continue
            with open(fileName, 'r') as aFile:
                # Flush md5 hash algorithm
                m = hashlib.md5()
                # Have to explicitly encode as all strings are unicode objects in Python 3 
                m.update(aFile.read(1024).encode('utf-8'))            
                hashValue = m.digest()
            if hashValue in hashes:
                x = hashes[hashValue]
                if type(x) is not trueType:
                    outFiles.append(hashes[hashValue])
                    hashes[hashValue] = True
                outFiles.append(fileName)
            else:
                hashes[hashValue] = fileName
            aFile.close()
        if len(outFiles):
            potentialDupes.append(outFiles)
            potentialCount = potentialCount + len(outFiles)
    del filesBySize
    
    print('Scanning for duplicates...')
    
    dupes = []
    for aSet in potentialDupes:
        outFiles = []
        hashes = {}
        for fileName in aSet:
            #print('Scanning file "%s"...' % fileName)
            with open(fileName, 'r') as aFile:
                m = hashlib.md5()
                while True:
                    r = aFile.read(4096).encode('utf-8')
                    if not len(r):
                        break
                    m.update(r)
            hashValue = m.digest()
            if hashValue in hashes:
                if not len(outFiles):
                    outFiles.append(hashes[hashValue])
                outFiles.append(fileName)
            else:
                hashes[hashValue] = fileName
        if len(outFiles):
            dupes.append(outFiles)
    i = 0
    goOn = 0
    choice = input("Found %i duplicates. (1) Show, (2) Delete, (3) Back to menu or (x) Exit\nINT>>> " % len(dupes))
    while goOn == 0:
        if choice == "1":
            for d in dupes:
                print('Original is %s' % d[0])
                for f in d[1:]:
                    print('Duplicate is %s' % f)
                choice = input("Found %i duplicates. (1) Show, (2) Delete, (3) Back to menu or (x) Exit\nINT>>> " % len(dupes))
        elif choice == "2":    
            for d in dupes:
                print('Original is %s' % d[0])
                for f in d[1:]:
                    i = i + 1
                    print('Deleting %s' % f)
                    os.remove(f)
            goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
            if goOn == "yes" or goOn == "y" or goOn == "yep":
                main_menu(myDir)
            else:
                exit()
        elif choice == "3":
            goOn = 1
            main_menu(myDir)
        elif choice == "x" or "X":
            exit()
        else:
            choice = input("Please enter 1, 2, 3 or x.")

def chunking(myDir):
    """Split large .txt file into chunks. Output is a folder of small .txt files.
    Size of the chunks is determined by user."""
    fileList, noFiles = list_textfiles(myDir)
    chunks = []
    try:
        chunkLength = int(input("How many words should each chunks be? Enter a number\n(suggested: 500-1000).\nINT>>> "))
    except ValueError:
        print("Please enter a number.")
        chunking(myDir)
    for filePath in fileList:
        chunk_counter = 0
        texts = split_text(filePath, chunkLength)
        for text in texts:
            chunk = {'text': text, 'number': chunk_counter, 'filename': filePath}
            chunks.append(chunk)
            chunk_counter += 1         
    output_dir = myDir + '-data_chunks'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)     
    for chunk in chunks:
        basename = os.path.basename(chunk['filename']).replace('.txt', '')
        fn = os.path.join(output_dir,
                          "{}{:04d}".format(basename, chunk['number']) + '.txt')
        with open(fn, 'w') as f:
            f.write(str(chunk['text']))
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()
                                
def stemmer(myDir):
    """Stems words. Creates directory in current directory with stemmed texts, or a single .csv file with top 100 stemmed words."""
    fileList, noFiles = list_textfiles(myDir)
    condOut = 0
    userFile = input("""Do you want [1] a .csv file with top stemmed counts or [2] .txt files with\nstemmed texts (useable as input for other functions)?\nINT>>> """)
    while condOut == 0:
        if userFile == "1":
            condOut = 1
        elif userFile == "2": 
            stemTXTDir = myDir + "-stem"
            if not os.path.exists(stemTXTDir):
                os.makedirs(stemTXTDir) 
            condOut = 2
        else:
            continue        
    condStop = 0
    userStop = input("Remove stopwords?\nY/N>>> ").lower()
    while condStop == 0:
        if userStop == "yes" or userStop == "y" or userStop == "yep":
            print("Removing stopwords...")
            condStop = 1
        elif userStop == "no" or userStop == "n" or userStop == "nope": 
            print("Leaving stopwords alone...")
            condStop = 2
        else:
            continue      
    totalStemList = []
    for filePath in fileList:
        fSmall = os.path.split(filePath)[1] 
        fName = os.path.splitext(fSmall)[0]
        tokens = get_tokens(filePath)
        filtered = [w for w in tokens if not w in stopwords.words('english')]
        stemmer = SnowballStemmer("english")
        if condStop == 1:
            stemmed = stem_tokens(filtered, stemmer)
        else:
            stemmed = stem_tokens(tokens, stemmer)
        text = ' '.join(stemmed)
        totalStemList.append(stemmed)
        if condOut == 2:
            with open(os.curdir + "/" + stemTXTDir + "/" + fName + "-stems" + ".txt", "a", newline='') as f:
                f.write(str(text))
    realList = []
    for l in totalStemList:
        for i in l:
            realList.append(i)
    totalCount = Counter(realList)
    top = totalCount.most_common(100)
    if condOut == 1:
        with open(myDir + "-stemmed" + ".csv", "a", newline='') as f:
            writer = csv.writer(f, delimiter= ",", quoting=csv.QUOTE_NONNUMERIC)
            for j in top:
                writer.writerow(j)
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()

def tagger(myDir):
    """POS tags words. Creates directory in current directory with POS tagged texts, or a .csv file
    with top 100 tagged words. Can also filter corpus by POS tags"""
    fileList, noFiles = list_textfiles(myDir)
    condOut = 0
    userFile = input("""Do you want [1] a .csv file with top tagged counts or [2] .txt files with\ntagged texts (useable as input for other functions)?\nINT>>> """)
    while condOut == 0:
        if userFile == "1":
            condOut = 1
        elif userFile == "2": 
            condOut = 2
        else:
            userFile = input("Please try again!")
            continue         
    condFilter = 0
    userFilter = input("Filter output for [1] none, [2] nouns, [3] pronouns or [4] verbs? \nTip: you can filter at a later stage by using the POS filter function.\nINT>>> ")
    while condFilter == 0:
        if userFilter == "1": 
            posTXTDir = myDir + "-POS"               
            if not os.path.exists(posTXTDir):
                os.makedirs(posTXTDir)  
            print("Filtering POS tokens...")
            condFilter = 1
        elif userFilter == "2": 
            posTXTDir = myDir + "-POS-nouns"               
            if not os.path.exists(posTXTDir):
                os.makedirs(posTXTDir)  
            print("Filtering for nouns...")
            condFilter = 2
        elif userFilter == "3": 
            posTXTDir = myDir + "-POS-pronouns"               
            if not os.path.exists(posTXTDir):
                os.makedirs(posTXTDir)  
            print("Filtering for pronouns...")
            condFilter = 3
        elif userFilter == "4": 
            posTXTDir = myDir + "-POS-verbs"               
            if not os.path.exists(posTXTDir):
                os.makedirs(posTXTDir)  
            print("Filtering for verbs...")
            condFilter = 4
        else:
            userFilter = input("Please try again!\nINT>>> ")
            continue
    condStop = 0
    if condFilter == 1:
        while condStop == 0:
            userStop = input("Remove stopwords?\nY/N>>> ").lower()
            if userStop == "yes" or userStop == "y":
                print("Removing stopwords...")
                condStop = 1
            if userStop == "no" or userStop == "n": 
                print("Leaving stopwords alone...")
                condStop = 2
            else:
                continue 
    totalPOSlist = []
    for filePath in fileList:
        fSmall = os.path.split(filePath)[1] 
        fName = os.path.splitext(fSmall)[0]
        tokens = get_tokens(filePath)
        if condStop == 1:
            filtered = [w for w in tokens if not w in stopwords.words('english')]
        else:
            filtered = tokens
        pos = nltk.pos_tag(filtered)
        posTokens = []
        if condFilter == 1:
            for tup in pos:
                posTokens.append(''.join(tup))
        elif condFilter == 2:
            for tup in pos:
                if tup[1] == "NN":
                    posTokens.append(''.join(tup))
        elif condFilter == 3:
            for tup in pos:
                if tup[1] == "JJ":
                    posTokens.append(''.join(tup))
        elif condFilter == 4:
            for tup in pos:
                if tup[1] == "VB" or tup[1] == "VBN" or tup[1] == "VBP" or tup[1] == "VBD":
                    posTokens.append(''.join(tup))
        posText = ' '.join(posTokens) 
        totalPOSlist.append(pos)    
        if condOut == 2:
            with open(os.curdir + "/" + posTXTDir + "/" + fName + "-POS" + ".txt", "w") as f:
                f.write(str(posText))
    realList = []
    for l in totalPOSlist:
        for i in l:
            realList.append(i)
    totalCount = Counter(realList)    
    totalTop = totalCount.most_common(100)
    if condOut == 1:        
        with open(myDir + "-POS" + ".csv", "a", newline='') as f:
            writer = csv.writer(f, delimiter= ",", quoting=csv.QUOTE_NONNUMERIC)
            for i in totalTop:
                writer.writerow(i)
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()

def pos_filter (myDir):
    """Filters POS-tagged folder of files by nouns, verbs or pronouns"""
    fileList, noFiles = list_textfiles(myDir)    
    testTokens = open(fileList[1]).read().split()   
    testCounter = 0
    for token in testTokens:
        if token[-2:] == "NN" or token[-2:] == "JJ" or token[-3:] == "VBN" or token[-3:] == "VBP" or token[-3:] == "VBD":
            testCounter += 1
    if testCounter < 5:
        print("The files in this folder are not POS-tagged! Exiting.")
        exit()
    realList = []
    userFilter = input("Filter output by [1] nouns, [2] pronouns or [3] verbs?\nINT>>> ")
    if userFilter != "1" and userFilter != "2" and userFilter != "3": 
        userFilter = input("Please try again!\nINT>>> ")
    elif userFilter == "1": 
        posTXTDir = myDir + "-nouns"               
        if not os.path.exists(posTXTDir):
            os.makedirs(posTXTDir)  
        for filePath in fileList:
            fSmall = os.path.split(filePath)[1] 
            fName = os.path.splitext(fSmall)[0]
            tokens = open(filePath).read().split()  
            realList = [word for word in tokens if word[-2:] == "NN"]
            with open(os.curdir + "/" + myDir + "-nouns" + "/" + fSmall, "w") as f:
                    f.write(' '.join(realList)) 
    elif userFilter == "2": 
        posTXTDir = myDir + "-pronouns"               
        if not os.path.exists(posTXTDir):
            os.makedirs(posTXTDir)  
        for filePath in fileList:
            fSmall = os.path.split(filePath)[1] 
            fName = os.path.splitext(fSmall)[0]
            tokens = open(filePath).read().split()  
            realList = [word for word in tokens if word[-2:] == "JJ"]        
            with open(os.curdir + "/" + myDir + "-pronouns" + "/" + fSmall, "w") as f:
                    f.write(' '.join(realList)) 
    elif userFilter == "3": 
        posTXTDir = myDir + "-verbs"          
        if not os.path.exists(posTXTDir):
            os.makedirs(posTXTDir)  
        for filePath in fileList:
            fSmall = os.path.split(filePath)[1] 
            fName = os.path.splitext(fSmall)[0]
            tokens = open(filePath).read().split()  
            allWords1 = [word for word in tokens if word[-2:] == "VB"]
            allWords2 = [word for word in tokens if word[-3:] == "VBN" or word[-3:] == "VBP" or word[-3:] == "VBD"]
            realList = allWords1 + allWords2
            with open(os.curdir + "/" + myDir + "-verbs" + "/" + fSmall, "w") as f:
                    f.write(' '.join(realList)) 
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()    

def lemmatizer(myDir):
    """Lemmatizes words. Creates directory in current directory with lemmatized texts, or a .csv file with top 100 lemmas."""
    fileList, noFiles = list_textfiles(myDir)
    condOut = 0
    userFile = input("""Do you want [1] a .csv file with top lemmas or [2] .txt files with\n lemmatized texts (useable as input for other functions)?\nINT>>> """)
    while condOut == 0:
        if userFile == "1":
            condOut = 1
        elif userFile == "2": 
            lemmaTXTDir = myDir + "-lemma"               
            if not os.path.exists(lemmaTXTDir):
                os.makedirs(lemmaTXTDir)  
            condOut = 2
        else:
            continue        
    condStop = 0
    userStop = input("Remove stopwords?\nY/N>>> ").lower()
    while condStop == 0:
        if userStop == "yes" or userStop == "y" or userStop == "yep":
            print("Removing stopwords...")
            condStop = 1
        elif userStop == "no" or userStop == "n" or userStop == "nope": 
            print("Leaving stopwords alone...")
            condStop = 2
        else:
            userStop = input("Please try again!\nY/N>>> ").lower()
            continue  
    totalLemmaList = []
    lmtzr = WordNetLemmatizer()
    for filePath in fileList:
        fSmall = os.path.split(filePath)[1] 
        fName = os.path.splitext(fSmall)[0]
        tokens = get_tokens(filePath)
        if condStop == 1:
            filtered = [w for w in tokens if not w in stopwords.words('english')]
        if condStop == 2:
            filtered = tokens
        lemmaList = [lmtzr.lemmatize(w) for w in filtered]
        lemmaString = ' '.join(lemmaList)
        totalLemmaList.append(lemmaList)
        if condOut == 2:
            with open(os.curdir + "/" + lemmaTXTDir + "/" + fName + "-lemmas" + ".txt", "w") as f:
                f.write(str(lemmaString))
    realList = []
    for l in totalLemmaList:
        for i in l:
            realList.append(i)
    totalCount = Counter(realList)    
    totalTop = totalCount.most_common(100)
    if condOut == 1:        
        with open(myDir + "-lemma" + ".csv", "a", newline='') as f:
            writer = csv.writer(f, delimiter= ",", quoting=csv.QUOTE_NONNUMERIC)
            for i in totalTop:
                writer.writerow(i)
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit() 
    
def word_count(myDir):
    """Simple word count function. Prints separate and total wordcount to terminal"""
    fileList, noFiles = list_textfiles(myDir)
    print("Word counts per subcorpus")
    print('%-*s %s' % (20, "Subcorpus", "Frequency"))
    print("------------------------------")
    totalWordCount = 0
    listIndex = 0
    myDict = {}
    myList = []     
    # Starts the loop that will put all the authors/subcorpora in their separate lists inside myDict
    for filePath in fileList:
        fName = filePath.split("/")[-1].split("-")[0]        
        if fName not in myList:
            author = fileList[listIndex].split("/")[-1].split("-")[0]
            myDict[fName] = listFromAuthor(author, fileList)        
            myList.append(fName)
        listIndex += 1
        myWordCounter = 0
    tokenDict = {}
    for key, values in myDict.items():
        authorTokens = []
        for value in values:
            tokenCounter = 0
            tokens = get_tokens(value)
            tokenCounter += len(tokens)    
            totalWordCount += len(tokens)
            authorTokens.append(tokenCounter)
        tokenDict[key] = sum(authorTokens)
    for k,v in sorted(tokenDict.items()):
        print('%-*s %i' % (20, k, v))     
    print("\nTotal word count")
    print(str(totalWordCount) + "\n")
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()
        
def top_words(myDir):
    """finds top N words, based on user input"""
    inp = input("How many top words should I find? \nINT>>> ")
    try:
       val = int(inp)
    except ValueError:
       print("Please enter a number!")
       top_words(myDir)
    condCSV = 0
    while condCSV == 0:
        userAns = input("Do you want a .csv file with the top words?\nY/N>>> ").lower()
        if userAns == "no" or userAns == "n" or userAns == "nope":
            condCSV = 1
        elif userAns == "yes" or userAns == "y" or userAns == "yep":
            print("Creating .csv...")
            outFile = "top_words.csv"
            f = open(outFile, "a", newline='')
            writer = csv.writer(f, delimiter= ",", quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow( ("word", "count") )    
            condCSV = 2
        else:
            continue
    allWords = []
    fileList, noFiles = list_textfiles(myDir)
    tokensCond = 0
    userTokens = input("Should I POS-tag the tokens first?\nY/N>>> ").lower()
    while tokensCond == 0:
        if userTokens == "yes" or userTokens == "y" or userTokens == "yep":
            print("POS tagging...")
            tokensCond = 1
        elif userTokens == "no" or userTokens == "n" or userTokens == "nope": 
            print("Using regular tokens...")
            tokensCond = 2
        else:
            userTokens = input("Please answer yes or no!\nY/N>>> ").lower()   
    if tokensCond == 1:
        for filePath in fileList:
                words = get_POS_tokens(filePath)
                allWords.extend(words)
    elif tokensCond == 2:
        for filePath in fileList:
            words = get_tokens(filePath)
            allWords.extend(words)
    posCond = 0 # This will hold the type of POS we're after
    if tokensCond == 1:    
        userPOS = input("Do you want to filter for [1] nouns, [2] pronouns, or [3] verbs?\nINT>>> ")
        valid = ["1", "2", "3"]
        if userPOS in valid:
            posCond = int(userPOS)
        else:
            userPOS = input("Please try again!\nINT>>> ")
    if posCond == 0:
        allWords = [word for word in allWords]
    elif posCond == 1:
        print("Looking for nouns...")
        allWords = [word[:-2] for word in allWords if word[-2:] == "NN"]
    elif posCond == 2:
        print("Looking for pronouns...")
        allWords = [word[:-2] for word in allWords if word[-2:] == "JJ"]        
    elif posCond == 3:
        print("Looking for verbs...")
        allWords1 = [word[:-2] for word in allWords if word[-2:] == "VB"]
        allWords2 = [word[:-3] for word in allWords if word[-3:] == "VBN" or word[-3:] == "VBP" or word[-3:] == "VBD"]
        allWords = allWords1 + allWords2
    condStop = 0
    if posCond == 0:
        inp2 = input("Remove stopwords? (EN only)\nY/N>>> ").lower()
        while condStop == 0:
            if inp2 == "no" or inp2 == "n":
                print("Leaving stopwords alone...")
                condStop = 1
            elif inp2 == "yes" or inp2 == "y":
                print("Removing stopwords...")
                stopWords = set(stopwords.words('english'))
                allWords = [w for w in allWords if not w in stopWords]
                condStop = 2    
            else:
                inp2 = input("please enter 'yes' or 'no'!\nY/N>>> ").lower()
                continue    
    fdist = nltk.FreqDist(allWords)        
    # Show the top N words in the list, with counts
    print("TOP N WORDS IN CORPUS")
    print('%-*s %s' % (20, "Word", "Frequency"))
    print("------------------------------")
    for word, frequency in fdist.most_common(int(inp)):
        print('%-*s %d' % (20, word, frequency))        
        if condCSV == 2:
            with open(outFile, "a", newline='') as f:
                writer.writerow( (word, frequency) )
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()

def word_find(myDir):
    """finds word of choice; generates a .csv file with (relative) frequencies,
    as well as two plots with relative word frequencies and word frequency details"""
    myWord = input("What word should I look for?\nSTR>>> ").lower()    
    cond = 0
    userFile = input("Do you want a .csv file with (relative) frequencies of the word?\nY/N>>> ").lower()
    while cond == 0:
        if userFile == "yes" or userFile == "y" or userFile == "yep":
            print("Creating .csv file...")
            outFile = "relative_word_freq-" + myWord + ".csv"
            f = open(outFile, "a", newline='')
            writer = csv.writer(f, delimiter= ",", quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow( ("filename", "wordcount", "word: %s" % myWord, "relFreq: %s" % myWord, "relFreqTotal: %s" % myWord) )    
            cond = 2
        elif userFile == "no" or userFile == "n" or userFile == "nope":
            cond = 1
        else:
            userFile = input("Please try again!\nY/N>>> ").lower()
            continue
    totalMyWord = 0
    totalWords = 0
    relFreqList = []
    relFreqTotalList = []
    fileList, noFiles = list_textfiles(myDir)
    listIndex = 0
    myDict = {}
    myList = []     
    allWords = []
    # Starts the first loop to get total word count and total user word count, which we'll use in the next loop.
    # This loop also serves to put all the authors/subcorpora in their separate lists inside myDict
    print("Tracing word frequencies...")
    for filePath in fileList:
        myWordCounter = 0
        words = get_tokens(filePath)  
        allWords.extend(words)
        wordCount = len(words)
        totalWords += wordCount
        for w in words:
            if myWord == w:
                myWordCounter += 1
        totalMyWord += myWordCounter
        fName = filePath.split("/")[-1].split("-")[0]
        if fName.endswith('.txt'):
            fName = fName.replace('.txt','')
        if fName not in myList:
            author = fileList[listIndex].split("/")[-1].split("-")[0]
            myDict[fName] = listFromAuthor(author, fileList)        
            myList.append(fName)
        listIndex += 1
    # Sort the dict (otherwise it's unordered and doesn't line up with the lists we've been creating) 
    oDict = OrderedDict(sorted(myDict.items(), key=lambda t: t[1]))
    # Start second loop to get relative frequencies per author
    relFreqDict = {}
    for key, values in oDict.items():
        myWordCounter = 0
        authorWordCount = 0
        for value in values:
            fSmall = os.path.split(value)[1] 
            fName = os.path.splitext(fSmall)[0]
            words = get_tokens(value)
            wordCount = len(words)
            authorWordCount += wordCount
            for w in words:
                if myWord == w:
                    myWordCounter += 1
        relFreq = ((myWordCounter / authorWordCount) * 100)  
        relFreqDict[key] = relFreq
        relFreqList.append(relFreq)
        relFreqTotal = ((myWordCounter / totalMyWord) * 100)
        relFreqTotalList.append(relFreqTotal)            
        # Create output file if user wants it
        if cond == 2:
            with open(outFile, "a", newline='') as f:
                writer.writerow( (value, wordCount, myWordCounter, relFreq) )
    # Print total ocurrances of myWord
    print('The word ' + '"' + myWord + '"' + ' appears ' + str(totalMyWord) + ' times in total.')
    ## Print place of word in freq dist
    fdist = nltk.FreqDist(allWords)        
    for i,j in enumerate(fdist.most_common(10000)):
        for word in j:
            if word == myWord:
                if i == 0:
                    print("It is the " + str(i+1) + "st most frequent word in the corpus.") 
                if i == 1:
                    print("It is the " + str(i+1) + "nd most frequent word in the corpus.") 
                if i == 2:
                    print("It is the " + str(i+1) + "rd most frequent word in the corpus.") 
                if i > 2:
                    print("It is the " + str(i+1) + "th most frequent word in the corpus.") 
         
    # Output figure 1: the search word normalized to the total no. of words in the subcorpus
    print("Generating output figure 1...")
    fig, ax = plt.subplots()
    N = len(relFreqList)
    x = np.arange(1, N+1)
    y = [num for num in relFreqList]
    labels = [s for s in myList]
    width = 1
    bar1 = plt.bar(x, y, width, color="lightcoral")
    plt.ylabel("relative frequency")
    plt.xticks(x - 1 + width, labels, rotation='30')
    # Set 'minor ticks' so that they are located halfway between the major ticks.
    # First, hide major tick labels
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    # Then, customize minor tick labels
    ax.xaxis.set_minor_locator(ticker.FixedLocator(linspace(1.5, 100.5, num=100)))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(labels))
    # Auto-limiting graph to the size of the plot
    plt.xlim(1, len(myList) + 1) 
    plt.title("Word: %s" % myWord + ", normalized to total no. of words in subcorpus", fontsize=15)    
    plt.show()
    
    # Output figure 2: the search word normalized to the total no. of that word in the whole corpus    
    print("Generating output figure 2...")
    labels = [s for s in myList]
    sizes = [num for num in relFreqTotalList]
    colors = ["lightcoral", "yellowgreen", "gold", "lightskyblue"]
    ## if we want, we could visually 'explode' the largest number
    #explodeList = []
    #for i in relFreqTotalList:
    #    if i != max(relFreqTotalList):
    #        explodeList.append(0)
    #    elif i == max(relFreqTotalList):
    #        explodeList.append(0.1)
    #explode = tuple(explodeList)
    ## If we're adding the explode function, we should add explode=explode in the next param
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=False, startangle=90)
    # Draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0,0),0.3,color='black', fc='white',linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.title("Word: '%s'" % myWord + ", normalized to frequency of '%s'" % myWord + " in total corpus", fontsize=15, y=1.08)    
    plt.show()    
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()

def find_concordances(myDir):
    """Prints concordances of a chosen word, iterating alphabetically or randomly through the corpus"""
    # Need function to ask how many surrounding words the user wants to show.. Too few as is.
    
    import random 
    fileList, noFiles = list_textfiles(myDir)
    userGram = 0
    userWord = input("Which word do you want to see the concordance of?\nSTR>>> ").lower()
    while userGram == 0:
        split = userWord.split(' ')        
        if len(split) < 2:
            userGram = 1
        if len(split) == 2:
            print("Looking for bigrams...")
            userGram = 2
        if len(split) == 3:
            print("Looking for trigrams...")
            userGram = 3
        if len(split) > 3:
            userWord = input("Please choose word, bigram or trigram!\nSTR>>> ").lower()
    sortCond = 0
    userSort = input("Should I iterate through the subcorpora [1] alphabetically or [2] randomly?\nINT>>> ")
    while sortCond == 0:
        valid = ["1", "2"]
        if userSort in valid:
            if userSort == "2":
                random.shuffle(fileList)
            sortCond = 1 
        else:
            userSort = input("Please try again.\nINT>>> ")
    if userGram == 1:
        for filePath in fileList:
            tokens = get_tokens(filePath)    
            x = None
            if userWord in tokens:  
                text = nltk.Text(tokens)
                print(str(filePath))
                print(text.concordance(userWord))
                x = input("Press ENTER to continue, M for main menu, or X to exit... \n").lower()
                if x == "x":
                    exit()
                elif x == "m":
                    main_menu(myDir)
                elif x != None:
                    continue
    elif userGram == 2:
        fakeUserWord = userWord.replace(" ", "_")
        for filePath in fileList:
            with open(filePath, 'r') as f:
                text = f.read()
            lowers = text.lower()
            no_punctuation = ''.join(ch for ch in lowers if category(ch)[0] != 'P')
            no_diacritics = ''.join(c for c in unicodedata.normalize('NFD', no_punctuation) if unicodedata.category(c) != 'Mn')
            string = no_diacritics.replace(userWord, userWord.replace(" ", "_"))
            if fakeUserWord in string:
                tokens = string.split()
                text = nltk.Text(tokens)
                print(str(filePath))
                print(text.concordance(fakeUserWord))
                x = input("Press ENTER to continue, M for main menu, or X to exit... \n").lower()
                if x == "x":
                    exit()
                elif x == "m":
                    main_menu(myDir)
                elif x != None:
                    continue
    elif userGram == 3:
        fakeUserWord = userWord.replace(" ", "_")
        for filePath in fileList:
            with open(filePath, 'r') as f:
                text = f.read()
            lowers = text.lower()
            if userWord in lowers:
                no_punctuation = ''.join(ch for ch in lowers if category(ch)[0] != 'P')
                no_diacritics = ''.join(c for c in unicodedata.normalize('NFD', no_punctuation) if unicodedata.category(c) != 'Mn')
                string = no_diacritics.replace(userWord, userWord.replace(" ", "_"))
                tokens = string.split()
                text = nltk.Text(tokens)
                print(str(filePath))
                print(text.concordance(fakeUserWord))
                x = input("Press ENTER to continue, M for main menu, or X to exit... \n").lower()
                if x == "x":
                    exit()
                elif x == "m":
                    main_menu(myDir)
                elif x != None:
                    continue   
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()

def find_clusters(myDir):
    """Finds cluster based on word of choice. User selects bi- or trigram."""
    fileList, noFiles = list_textfiles(myDir)
    userWord = input("What word do you want to see the lexical cluster of?\nSTR>>> ").lower()    
    userCond = 0
    while userCond == 0:
        split = userWord.split(' ')        
        if len(split) > 1:
            userWord = input("Please choose 1 word!\nSTR>>> ").lower()
        else:
            userCond = 1
    gramCond = 0
    userGram = input("Do you want [1] bigrams or [2] trigrams?\nINT>>> ")
    valid = ["1", "2"]
    while gramCond == 0:
        if userGram in valid:
            gramCond = int(userGram)
        else:
            userGram = input("Please try again.\nINT>>> ")
    noCond = 0
    userNo = input("How many top clusters (1-100) should I find?\nINT>>> ")
    while noCond == 0:
        valid = [str(n) for n in range(1,100)]
        if userNo in valid:
            noCond = int(userNo)
        else:
            userNo = input("Please try again.\nINT>>> ")
    totalTokens = []
    for filePath in fileList:
        tokens = get_tokens(filePath)
        if userWord in tokens:
            totalTokens.extend(tokens)
    if gramCond == 1:
        biTokens = bigrams(totalTokens)
        # count which ones appear most frequently
        biList = [item for item in biTokens if userWord in item]
        biCounter = Counter(biList)
        biCommon = biCounter.most_common(noCond)        
        print('%-*s %-*s %s' % (10, "Word 1", 10, "Word 2", "Freq"))
        for words, freq in biCommon:            
            print('%-*s %-*s %s' % (10, str(words[0]), 10, str(words[1]), freq))  
    else:
        triTokens = trigrams(totalTokens)
        triList = [item for item in triTokens if userWord in item]
        triCounter = Counter(triList)
        triCommon = triCounter.most_common(noCond)        
        print('%-*s %-*s %-*s %s' % (10, "Word 1", 10, "Word 2", 10, "Word 3", "Freq"))
        for words, freq in triCommon:            
            print('%-*s %-*s %-*s %s' % (10, str(words[0]), 10, str(words[1]), 10, str(words[2]), freq))          
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()

def lexical_variety(myDir):
    """Calculates and visualizes mean word use and TTF scores. If input is a hyphened series of chunks,
    the output is organized per subcorpus"""
    fileList, noFiles = list_textfiles(myDir)
    author = fileList[0].split("/")[-1]
    if "-" in author:        # We could ask the user for a different escape char at te start too
        print("Found split subcorpora in folder. Will concatenate for evaluation.") 
    else:
        print("Found unsplit subcorpora in folder. If the files are big, consider splitting them \nusing Corporeal's chunking function.")
    cond = 0
    userFile = input("Do you want a .csv file with means and TTF scores per file?\nY/N>>> ").lower()
    while cond == 0:
        if userFile == "yes" or userFile == "y" or userFile == "yep":
            print("Creating .csv file...")
            outFile = "lexical_variety_split.csv"
            f = open(outFile, "a", newline='')
            writer = csv.writer(f, delimiter= ",", quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow( ("filename", "mean", "TTF") )    
            cond = 2
        elif userFile == "no" or userFile == "n" or userFile == "nope":
            cond = 1
        else:
            userFile = input("Please try again!\nY/N>>> ").lower()
    totalWordCounter = 0
    totalTypeCounter = 0
    allMeans = {}
    allTTR = {}
    tokensCond = 0
    userTokens = input("Do you want to use [1] regular tokens or [2] POS-tagged tokens?\nINT>>> ")
    valid = ["1", "2"]
    if userTokens in valid:
        tokensCond = int(userTokens)
    else:
        print("Please try again.")
    totalWordCounter = 0
    totalTypeCounter = 0
    allMeans = {}
    allTTR = {}
    listIndex = 0
    myDict = {}
    myList = []     
    # Starts the loop that will put all the authors/subcorpora in their separate lists inside myDict
    for filePath in fileList:
        fName = filePath.split("/")[-1].split("-")[0]
        if fName.endswith('.txt'):
            fName = fName.replace('.txt','')
        if fName not in myList:
            author = fileList[listIndex].split("/")[-1].split("-")[0]
            myDict[fName] = listFromAuthor(author, fileList)        
            myList.append(fName)
        listIndex += 1
    meansDict = {}
    TTRDict = {}
    for key, values in myDict.items():
        authorTTR = []
        authorMean = []
        for value in values:
            tokenCounter = 0
            typeCounter = 0
            fSmall = os.path.split(value)[1] 
            fName = os.path.splitext(fSmall)[0]        
            if tokensCond == 1:
                tokens = get_tokens(value)
            else:
                tokens = get_POS_tokens(value) 
            tokenCounter += len(tokens)    
            typeCounter = len(set(tokens))
            totalWordCounter += len(tokens)
            totalTypeCounter += len(set(tokens))
            mean = tokenCounter / typeCounter
            TTR = typeCounter / tokenCounter
            allTTR[fName] = TTR 
            authorMean.append(mean)
            authorTTR.append(TTR)
        meansDict[key] = sum(authorMean) / len(authorMean)
        TTRDict[key] = sum(authorTTR) / len(authorTTR)
    print("Lexical variety per subcorpus")
    print('%-*s %-*s %s' % (30, "Corpus Name", 20, "Mean Word Freq", "Type-Token Ratio"))
    for (k,v), (k2,v2) in zip(meansDict.items(), TTRDict.items()):
        print('%-*s %-*f %f' % (30, k, 20, v, v2))     
        if cond == 2:
            with open(outFile, "a", newline='') as f:
                writer.writerow( (k, v, v2) )
    # Calculate total means
    totalMeans = totalWordCounter / totalTypeCounter
    # Dict comprehension to normalize values, recalculating meansDict
    meansDict = {key:value-totalMeans for key, value in meansDict.items()}

    # Plotting means, sorted by values
    fig, ax = plt.subplots()
    sortedMeans = OrderedDict(sorted(meansDict.items(), key=lambda t: t[1]))
    N = len(sortedMeans)
    x = np.arange(1, N+1)
    y = [num for num in sortedMeans.values()]
    labels = sorted(meansDict, key=meansDict.get, reverse=False)
    width = 1
    bar1 = plt.bar(x, y, width, color="lightcoral")
    plt.ylabel("mean word use")
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator(linspace(1.5, 100.5, num=100)))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(labels))
    plt.xlim(1, len(sortedMeans) +1)
    plt.title("Lexical repetitiveness by normalized mean word use per subcorpus, ordered by value", fontsize=14, y=1.03)    
    plt.show()
        
    # Plotting TTF, sorted by values
    fig, ax = plt.subplots()
    sortedTTR = OrderedDict(sorted(TTRDict.items(), key=lambda t: t[1]))
    N = len(sortedTTR)
    x = np.arange(1, N+1)
    y = [num for num in sortedTTR.values()]
    labels = sorted(TTRDict, key=TTRDict.get, reverse=False)
    width = 1
    bar1 = plt.bar(x, y, width, color="lightcoral")
    plt.ylabel("TTR")
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator(linspace(1.5, 100.5, num=100)))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(labels))
    plt.xlim(1, len(sortedTTR) +1)
    plt.title("Lexical variety by TTR score per subcorpus, ordered by value", fontsize=14, y=1.03)    
    plt.show()
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()

def distinctive(myDir):
    """Compare the average rate at which words are used in (A) the user-defined subcorpus and (B) the
    rest of the corpus. We calculate the difference between the rates to calculate their
    distinciveness in the subcorpus that the user enters"""
    fileList, noFiles = list_textfiles(myDir)
    userName = input("Please enter name of subcorpus you want to see distinctive features of\nSTR>>> ").lower()  
    testDir = []
    for i in fileList:
        fSmall = os.path.split(i)[1] 
        fName = os.path.splitext(fSmall)[0]
        try: 
            i = fName.split("-")[0]
            testDir.append(i)
        except:
            testDir.append(i)
    if userName not in testDir:
        print("That name does not seem to exist in this folder.. Try again.")
        distinctive(myDir)
    userNo = input("How many distinctive features should I find?\nINT>>> ")
    try:
       val = int(userNo)
    except ValueError:
       print("Please enter a number!")
       distinctive(myDir)    
    condCSV = 0
    while condCSV == 0:
        userAns = input("Do you want a .csv file with the top words?\nY/N>>> ").lower()
        if userAns == "no" or userAns == "n" or userAns == "nope":
            condCSV = 1
        elif userAns == "yes" or userAns == "y" or userAns == "yep":
            print("Creating .csv file...")
            outFile = "distinctive_words-" + userName + ".csv"
            # We could add a condition here that if the file exists, ask user to overwrite            
            f = open(outFile, "a", newline='')
            writer = csv.writer(f, delimiter= ",", quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow( ("subcorpus = " + userName,) )
            writer.writerow( ("word", "keyness") )    
            condCSV = 2
        else:
            print("Please try again!")
            continue  
    vectorizer = CountVectorizer(input='filename') # Input expects a list of filenames
    dtm = vectorizer.fit_transform(fileList)
    vocab = np.array(vectorizer.get_feature_names())
    dtm = dtm.toarray()
    rates = 1000 * dtm / np.sum(dtm, axis=1, keepdims=True)
    userNameIndices = []
    otherNamesIndices = []
    fileList, noFiles = list_textfiles(myDir)
    for index, fn in enumerate(fileList):
        if userName in fn:
            userNameIndices.append(index)
        else:
            otherNamesIndices.append(index)
    userNameRates = rates[userNameIndices, :]
    otherNamesRates = rates[otherNamesIndices, :]        
    userNameRatesAvg = np.mean(userNameRates, axis=0)
    otherNamesRatesAvg = np.mean(otherNamesRates, axis=0)
    distinctiveIndices = (userNameRatesAvg * otherNamesRatesAvg) == 0
    ranking = np.argsort(userNameRatesAvg[distinctiveIndices] + otherNamesRatesAvg
                     [distinctiveIndices])[::-1] # from highest to lowest; [::-1] reverses order
    dtm = dtm[:, np.invert(distinctiveIndices)]
    rates = rates[:, np.invert(distinctiveIndices)]
    vocab = vocab[np.invert(distinctiveIndices)]
    # recalculate variables that depend on rates
    userNameRates = rates[userNameIndices, :]
    otherNamesRates = rates[otherNamesIndices, :]
    userNameRatesAvg = np.mean(userNameRates, axis=0)
    otherNamesRatesAvg = np.mean(otherNamesRates, axis=0)
    keyness = np.abs(userNameRatesAvg - otherNamesRatesAvg)
    ranking = np.argsort(keyness)[::-1]  # from highest to lowest; [::-1] reverses order
    ratesAvg = np.mean(rates, axis=0)
    keyness = np.abs(userNameRatesAvg - otherNamesRatesAvg) / ratesAvg
    ranking = np.argsort(keyness)[::-1]  # from highest to lowest; [::-1] reverses order.
    topKeyness = [i for i in keyness[ranking][0:int(userNo)]]
    topVocab = [i for i in vocab[ranking][0:int(userNo)]]
    d = dict(zip(topVocab, topKeyness))
    print("TOP " + userNo + " DISTINCTIVE WORDS PER SUBCORPUS '" + str(userName).upper() + "' BY COMPARING AVG RATES")     
    print("%-*s %s" % (20, "Word", "Keyness"))
    for k, v in d.items():
        print("%-*s %f" % (20, k, v))
        if condCSV == 2:
            with open(outFile, "a", newline='') as f:
                for k, v in d.items():
                    writer.writerow( (k,v) )  
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()   

def compare_POS(myDir):
    """Compare the parts of speech of a chosen word (e.g., see if 'like' is used as a verb or noun"""
    fileList, noFiles = list_textfiles(myDir)
    # Short loop to check if input folder is POS-tagged
    checkSum = 0
    for filePath in fileList[0:5]:
        with open(filePath, 'r') as f:
            text = f.read()
            no_punctuation = ''.join(ch for ch in text if category(ch)[0] != 'P')
            no_diacritics = ''.join(c for c in unicodedata.normalize('NFD', no_punctuation)
                      if unicodedata.category(c) != 'Mn')
            tokens = no_diacritics.split()
        for token in tokens:        
            if "NN" in token or "JJ" in token or "VB" in token:
                checkSum += 1
    if checkSum < 5:
        print("It looks like the input folder is not POS-tagged properly. Use the Corporeal tagger to tag the corpus first.")
        goOn = input("Back to main menu?\nY/N>>> ").lower()
        if goOn == "yes" or goOn == "y" or goOn == "yep":
            main_menu(myDir)
        else:
            exit()
    userWord = input("Which word do you want to compare the POS of?\nSTR>>> ").lower()
    tokenList = []
    print("Comparing tokens...")
    # Tokens in text files are appended by POS tag. We remove the POS tag (they're in uppercase)
    # to trace the user word. We can't use our get_tokens function as it lowercases everything
    for fn in fileList:        
        with open(fn, 'r') as f:
            text = f.read()
            no_punctuation = ''.join(ch for ch in text if category(ch)[0] != 'P')
            no_diacritics = ''.join(c for c in unicodedata.normalize('NFD', no_punctuation)
                      if unicodedata.category(c) != 'Mn')
            tokens = no_diacritics.split()        
        for token in tokens:
            testToken = re.sub(r'([A-Z]){2}', "", token)
            if userWord == testToken:          
                tokenList.append(token)
    # Count the times user word appears in different forms
    countedTokenList = Counter(tokenList)
    # Output comparison
    for key, value in countedTokenList.items():
        print(key, value)
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()
    
def euclidian(myDir):
    """Calculates Euclidian distances between subcorpora.
    If corpora are split, they will be concatenated before analysis."""
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.manifold import MDS
    from scipy.cluster.hierarchy import ward, dendrogram
    import matplotlib.pyplot as plt
    fileList, noFiles = list_textfiles(myDir)    
    author = fileList[0].split("/")[-1]
    if "-" in author:        # We could ask the user for a different escape char at te start too
        print("Found split subcorpora in folder. Will concatenate for evaluation.") 
    else:
        print("Found unsplit subcorpora in folder.")
    listIndex = 0
    myDict = {} # The dict to contain sorted filenames
    myList = [] # Just some list
    # Loop that will put all the authors and associated filenames inside myDict
    for filePath in fileList:
        fName = filePath.split("/")[-1].split("-")[0]        
        if fName.endswith('.txt'):
            fName = fName.replace('.txt','')
        if fName not in myList:
            author = fileList[listIndex].split("/")[-1].split("-")[0]
            myDict[fName] = listFromAuthor(author, fileList)        
            myList.append(fName)
        listIndex += 1
        myWordCounter = 0
    # Append all texts together per author and use those
    tokensDict = {}
    for key, values in myDict.items():
        tokensList = []
        for v in values:
            t = get_tokens(v)
            tokensList.append(t)
        # Makeshift way to append items in nested lists to 1 other list
        realList = []
        for l in tokensList:
            for i in l:
                realList.append(i)
        # Transform into str
        tokens = ' '.join(realList)       
        tokensDict[key] = tokens
    sortedDict = OrderedDict(sorted(tokensDict.items(), key=lambda t: t[0]))    
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(sortedDict.values())
    vocab = vectorizer.get_feature_names()  # a list    
    dtm = dtm.toarray()  # convert to a regular array
    vocab = np.array(vocab)
    dist = euclidean_distances(dtm)
    #2D plot
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    names = [i for i in sortedDict.keys()]
    for x, y, name in zip(xs, ys, names):
        color = 'lightcoral'
        plt.scatter(x, y, c=color)
        plt.text(x, y, name)
    plt.title("2D Euclidian distances between subcorpora")
    plt.show()
    
    #dendogram
    linkage_matrix = ward(dist)
    dendrogram(linkage_matrix, orientation="right", labels=names);
    plt.tight_layout()  # fixes margins
    plt.title("Dendogram using Euclidian distances between subcorpora")
    plt.show()
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()
    
def cosine(myDir):
    """Calculates Cosine distances between subcorpora.
    If corpora are split, they will be concatenated before analysis."""
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.manifold import MDS
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.cluster.hierarchy import ward, dendrogram
    from sklearn.feature_extraction.text import TfidfVectorizer
    fileList, noFiles = list_textfiles(myDir)
    author = fileList[0].split("/")[-1]
    if "-" in author:        # We could ask the user for a different escape char at te start too
        print("Found split subcorpora in folder. Will concatenate for evaluation.") 
    else:
        print("Found unsplit subcorpora in folder.")
    listIndex = 0
    myDict = {} # The dict to contain sorted filenames
    myList = [] # Just some list
    # Loop that will put all the authors and associated filenames inside myDict
    for filePath in fileList:
        fName = filePath.split("/")[-1].split("-")[0]        
        if fName.endswith('.txt'):
            fName = fName.replace('.txt','')
        if fName not in myList:
            author = fileList[listIndex].split("/")[-1].split("-")[0]
            myDict[fName] = listFromAuthor(author, fileList)        
            myList.append(fName)
        listIndex += 1
        myWordCounter = 0
    # Append all texts together per author and use those
    tokensDict = {}
    for key, values in myDict.items():
        tokensList = []
        for v in values:
            t = get_tokens(v)
            tokensList.append(t)
        # Makeshift way to append items in nested lists to 1 other list
        realList = []
        for l in tokensList:
            for i in l:
                realList.append(i)
        # Transform into str
        tokens = ' '.join(realList)       
        tokensDict[key] = tokens        
    sortedDict = OrderedDict(sorted(tokensDict.items(), key=lambda t: t[0]))    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(sortedDict.values())
    dtm = tfidf.toarray()
    # Cosine similarity is a measure of similarity so we need to 'flip' the measure so that a
    # larger angle receives a larger value. The distance measure derived from cosine similarity
    # is thus one minus the cosine similarity between two vectors.
    dist = 1 - cosine_similarity(dtm)
    
    # 2d plot
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    names = [i for i in sortedDict.keys()]
    for x, y, name in zip(xs, ys, names):
        color = 'blue'
        plt.scatter(x, y, c=color)
        plt.text(x, y, name)  
    plt.title("2D cosine distances between subcorpora")
    plt.show()
    
    #dendogram
    linkage_matrix = ward(dist)
    dendrogram(linkage_matrix, orientation="right", labels=names);
    plt.tight_layout()  # fixes margins
    plt.title("Dendogram using cosine distances between subcorpora")
    plt.show()
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()



# ----- UNDER CONSTRUCTION -----

def discourse_freq(myDir):
    """User enters 5 words, function calculates the mean of their relative frequencies"""
    
    goOn = 0
    while goOn == 0:
        try:
            wordList = input("Pick 5 words separated by space\nSTR>>> ").split(" ")
            print("Using words: " + str(wordList))
        except ValueError:
            continue
        goOn = 1
    discourse = input("Give a name to the discourse (for graph output)\nSTR>>> ")

    # Now replicate word find function but loop over all words in wordList, finally
    # calculating means over relative frequencies and offset this against the total word count

  #  for word in wordList:
    

    totalMyWord = 0
    totalWords = 0
    relFreqList = []
    relFreqTotalList = []
    fileList, noFiles = list_textfiles(myDir)
    listIndex = 0
    myDict = {}
    myList = []     
    allWords = []
    # Starts the first loop to get total word count and total user word count, which we'll use in the next loop.
    # This loop also serves to put all the authors/subcorpora in their separate lists inside myDict
    print("Tracing word frequencies...")
    for filePath in fileList:
        myWordCounter = 0
        words = get_tokens(filePath)  
        allWords.extend(words)
        wordCount = len(words)
        totalWords += wordCount
        for w in words:
            if myWord in w:
                myWordCounter += 1
        totalMyWord += myWordCounter
        fName = filePath.split("/")[-1].split("-")[0]
        if fName.endswith('.txt'):
            fName = fName.replace('.txt','')
        if fName not in myList:
            author = fileList[listIndex].split("/")[-1].split("-")[0]
            myDict[fName] = listFromAuthor(author, fileList)        
            myList.append(fName)
        listIndex += 1
    # Sort the dict (otherwise it's unordered and doesn't line up with the lists we've been creating) 
    oDict = OrderedDict(sorted(myDict.items(), key=lambda t: t[1]))
    # Start second loop to get relative frequencies per author
    relFreqDict = {}
    for key, values in oDict.items():
        myWordCounter = 0
        authorWordCount = 0
        for value in values:
            fSmall = os.path.split(value)[1] 
            fName = os.path.splitext(fSmall)[0]
            words = get_tokens(value)
            wordCount = len(words)
            authorWordCount += wordCount
            for w in words:
                if myWord == w:
                    myWordCounter += 1
        relFreq = ((myWordCounter / authorWordCount) * 100)  
        relFreqDict[key] = relFreq
        relFreqList.append(relFreq)
        relFreqTotal = ((myWordCounter / totalMyWord) * 100)
        relFreqTotalList.append(relFreqTotal)            


    # Add new calculation here: mean the relative frequencies of these words, which forms
    # the basis for the new output
    

         
    # Output figure 1: the search word normalized to the total no. of words in the subcorpus
    print("Generating output figure 1...")
    fig, ax = plt.subplots()
    N = len(relFreqList)
    x = np.arange(1, N+1)
    y = [num for num in relFreqList]
    labels = [s for s in myList]
    width = 1
    bar1 = plt.bar(x, y, width, color="lightcoral")
    plt.ylabel("relative frequency")
    plt.xticks(x - 1 + width, labels, rotation='30')
    # Set 'minor ticks' so that they are located halfway between the major ticks.
    # First, hide major tick labels
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    # Then, customize minor tick labels
    ax.xaxis.set_minor_locator(ticker.FixedLocator(linspace(1.5, 100.5, num=100)))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(labels))
    # Auto-limiting graph to the size of the plot
    plt.xlim(1, len(myList) + 1) 
    plt.title("Word: %s" % myWord + ", normalized to total no. of words in subcorpus", fontsize=15)    
    plt.show()
    
    # Output figure 2: the search word normalized to the total no. of that word in the whole corpus    
    print("Generating output figure 2...")
    labels = [s for s in myList]
    sizes = [num for num in relFreqTotalList]
    colors = ["lightcoral", "yellowgreen", "gold", "lightskyblue"]
    ## if we want, we could visually 'explode' the largest number
    #explodeList = []
    #for i in relFreqTotalList:
    #    if i != max(relFreqTotalList):
    #        explodeList.append(0)
    #    elif i == max(relFreqTotalList):
    #        explodeList.append(0.1)
    #explode = tuple(explodeList)
    ## If we're adding the explode function, we should add explode=explode in the next param
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=False, startangle=90)
    # Draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0,0),0.3,color='black', fc='white',linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.title("Word: '%s'" % myWord + ", normalized to frequency of '%s'" % myWord + " in total corpus", fontsize=15, y=1.08)    
    plt.show()    
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()


def similar_words(myDir):
    """Prints words that appear in similar contexts, iterating alphabetically or randomly through the corpus"""    
    import random 
    fileList, noFiles = list_textfiles(myDir)
    userWord = input("Which word do you want to see the similarity of?\nSTR>>> ").lower()
    sortCond = 0
    wordList = []
    for filePath in fileList:
        tokens = get_tokens(filePath)    
        if userWord in tokens:  
            text = nltk.Text(tokens)
            wordList.append(text.similar(userWord))
    #print(wordList)
    #for count, elem in sorted(((wordList.count(e), e) for e in set(wordList)), reverse=True):
    #    print('%s (%d)' % (elem, count))
    
                
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()
        

def common_contexts(myDir):
    """Prints words that appear in similar contexts, iterating alphabetically or randomly through the corpus"""    
    import random 
    fileList, noFiles = list_textfiles(myDir)
    userWord = input("Which word do you want to see the common context of?\nSTR>>> ").lower().split()
    print(userWord)
    sortCond = 0
    wordList = []
    for filePath in fileList:
        tokens = get_tokens(filePath)    
        if userWord[0] in tokens and userWord[1] in tokens:  
            text = nltk.Text(tokens)
            text.common_contexts(userWord)
    #print(wordList)
    #for count, elem in sorted(((wordList.count(e), e) for e in set(wordList)), reverse=True):
    #    print('%s (%d)' % (elem, count))
    
                
    goOn = input("Done! Back to main menu?\nY/N>>> ").lower()
    if goOn == "yes" or goOn == "y" or goOn == "yep":
        main_menu(myDir)
    else:
        exit()
        


def surrounding_phrases(myDir):
    """Basically a more targeted concordance function. User enters a word, function outputs the
    two sentences surrounding the word. This can be used to seek out specific reasons
    surrounding a topic of interest"""
    fileList, noFiles = list_textfiles(myDir)
    userWord = input("Which word do you want to see the surrounding phrases of?\nSTR>>> ").lower()
    print("Searching for word...")
    #for filePath in fileList:    
    #    sentences = nltk.sent_tokenize(filePath)
    #    tokens = [nltk.word_tokenize(sent) for sent in sentences]
    #    for token in tokens:           
    #        for word in token:            
    #            if userWord == word:
    #                # SOMETHING
    
if __name__ == '__main__':
    main()