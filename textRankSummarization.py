# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:19:05 2016

@author: Willy
"""

import io
import nltk
import networkx
import operator
from string import punctuation
from nltk.corpus import stopwords
from math import log10

def extractSent(numList, text):
    """
    Build the final summary output
    """
    summary = ""
    
    for n in numList:
        summary += (text[n] + ' ')

    return summary    
    
def extractSentNum(sortedSent):
    """
    Get the clean ordered sentences index number list
    """
    sentNum = []
    j = 0
    minLength = 3
    maxLength = 0

    if (len(sortedSent) < minLength):
        maxLength = len(sortedSent)
    else:
        maxLength = 3;
        
    while j < maxLength:
        sentNum.append(sortedSent[j][2])
        j+=1
        
    sentNum = sorted(sentNum, reverse=False)
    
    return sentNum

def textRank(graph):
    """
    TextRank algorithm
    """
    #Initialize default values
    alpha = 0.85
    inbound = 0
    outbound = 0
    maxIter = 100
    nIter = 0
    treshold = 0.0001
    lastScore = 0
    totalout = 0
    
    #Iterate until convergence is reached OR iteration limit reached
    while (nIter < maxIter):
        for n in graph:
            for pair in graph.edges(n):
                for pair2 in graph.edges(pair[1]):
                    outbound+= graph[pair2[0]][pair2[1]]["weight"]
                    totalout+=outbound
                inbound+= (graph[pair[0]][pair[1]]["weight"]/outbound)*graph.node[pair[1]]["nWeight"]
                outbound = 0
                totalout=0
            
            #Calculate TextRank score of node n
            graph.node[n]["nWeight"]= (1-alpha) + alpha * inbound
          
            inbound = 0
            
        #Check convergence    
        if nIter == 0:
            lastScore = graph.node[0]["nWeight"]
        else:
            if abs(graph.node[0]["nWeight"] - lastScore) < treshold:
                break
            lastScore = graph.node[0]["nWeight"]
        nIter+=1
    
    #Convert nodes to sentences list
    calculatedSent = []
    for n in graph:
        calculatedSent.append([graph.node[n]["nWeight"],graph.node[n]["string"],n])
    
    #Sort sentences list by TextRank score
    sortedSent = sorted(calculatedSent, key=operator.itemgetter(0), reverse=True)

    return sortedSent
    
    
def similarityCalc(pair,graph):
    """
    Calculate the similarity value between two sentence nodes
    """
    #Split sentence to words list
    firstSentWords = graph.node[pair[0]]["string"].split()
    secondSentWords = graph.node[pair[1]]["string"].split()
  
    #Counting common words in both sentences
    commonWords = len(set(firstSentWords) & set(secondSentWords))
    
    #Counting sentences length for normalizing
    firstSentLog = log10(len(firstSentWords))
    secondSentLog = log10(len(secondSentWords))
    
    normalizer = firstSentLog + secondSentLog
    if (normalizer == 0): 
        return 0
    else:
        return commonWords / (firstSentLog + secondSentLog)


def buildGraph(nodes):
    """
    Build a complete graph representation of the text document
    """
    #Initialize graph
    graph = networkx.Graph()
    for n,node in enumerate(nodes):
        graph.add_node(n, string = node, nWeight = 1)
    
    #Pair each nodes (create edges)
    pairedGraph = []
    for firstNode in graph:
        for secondNode in graph:
            if((firstNode != secondNode) & (firstNode < secondNode)): 
                pairedGraph.append([firstNode,secondNode])
    
                
    #Calculate edges' weight
    for pair in pairedGraph:
        edgeWeight = similarityCalc(pair,graph)
        
        if (edgeWeight!=0):
            graph.add_edge(pair[0], pair[1], weight=edgeWeight)
            
    return graph

def cleanStopWords(text):
    """
    Delete stopwords
    """
    text = ' '.join([word for word in text.split() if word not in (stopwords.words('english'))])
    
    return ''.join(c for c in text if c not in punctuation)
    
def clean(text):
    """
    Delete stopwords per sentence
    """
    sentenceToken = text
    
    for i, val in enumerate(sentenceToken):
        sentenceToken[i] = cleanStopWords(sentenceToken[i])

    return sentenceToken

def deletePunct(text):
    text = [w.replace('.', '') for w in text]
    text = [w.replace(',', '') for w in text]
    text = [w.replace(':', '') for w in text]
    text = [w.replace('!', '') for w in text]
    text = [w.replace('?', '') for w in text]
    text = [w.replace("'", '') for w in text]
    text = [w.replace('"', '') for w in text]
    text = [w.replace('-', ' ') for w in text]
    text = [w.replace('(', '') for w in text]
    text = [w.replace(')', ' ') for w in text]
    text = [w.replace('[', '') for w in text]
    text = [w.replace(']', ' ') for w in text]
    text = [w.replace('{', '') for w in text]
    text = [w.replace('}', ' ') for w in text]
   
    return text
    
def tokenize(text):
    """
    Divide text document to sentences list
    """
    engSentDetect = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizedText = engSentDetect.tokenize(text.strip())
                     
    return tokenizedText
    
def summarizeText(text):
    """
    Return a summarized form of the text document
    """
    #Detect and tokenize english sentences from text document
    tokenText = tokenize(text)  
    
    #Tokenized sentences without punctuations:
    noPunctText = list(tokenText)
    noPunctText = deletePunct(noPunctText)    
    
    #Tokenized sentences without stop words:
    cleanText = list(noPunctText)
    cleanText = clean(cleanText)    
    
    #Build graph from the sentences 
    graph = buildGraph(cleanText)
    
    #Employ the TextRank algorithm
    sortedSent = textRank(graph)
    
    #Get summary sentences index
    finalSentNum = extractSentNum(sortedSent)
    
    #Build summary
    summary = extractSent(finalSentNum,tokenText)
    
    return summary
    
reviewFile = io.open('review.txt','r')
text = reviewFile.read()
summary = summarizeText(text)
summaryFile = io.open('summary.txt', 'w',  encoding='utf-8')
summaryFile.write(unicode(summary))
summaryFile.close()
