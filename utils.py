# 2018 Su Wang (wangsu@google.com)

import os
import random
import torch
import numpy as np

from torch.autograd import Variable

class Indexer:
    """Word <-> Index mapper."""

    def __init__(self, specialTokenList=None):
        self.word2index = {}
        self.index2word = {}
        self.wordSet = set()
        self.size = 0
        if specialTokenList is not None:
            assert type(specialTokenList)==list
            for token in specialTokenList:
                self.get_index(token, add=True)

    def __repr__(self):
        return "The indexer currently has %d words" % self.size

    def get_word(self, index):
        return self.index2word[index] if index<self.size else 'UNK'

    def get_index(self, word, add=True):
        if add and word not in self.wordSet:
            self.word2index[word] = self.size
            self.index2word[self.size] = word
            self.wordSet.add(word)
            self.size += 1
        return self.word2index[word] if word in self.wordSet else self.word2index['UNK']

    def contains(self, word):
        return word in self.wordSet

    def add_sentence(self, sentence, returnIndices=True):
        indices = [self.get_index(word, add=True) for word in sentence.split()]
        return (indices,len(indices)) if returnIndices else None

    def add_document(self, docPath, returnIndices=True):
        with open(docPath, 'r') as doc:
            if returnIndices:
                indicesList, lengthList = [], []
                for line in doc:
                    indices,length = self.add_sentence(line,returnIndices)
                    if length<=0: continue # handle bad sentences in .txt.
                    indicesList.append(indices)
                    lengthList.append(length)
                return indicesList, lengthList
            else:
                for line in doc:
                    self.add_sentence(line,returnIndices=False)
                return None
    
    def to_words(self, indices):
        return [self.get_word(index) for index in indices]
    
    def to_sent(self, indices):
        return ' '.join(self.to_words(indices))
    
    def to_indices(self, words):
        return [self.get_index(word) for word in words]

class IndexerTransformer:
    """Word <-> Index mapper."""

    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.wordSet = set()
        self.size = 0

    def __repr__(self):
        return "The indexer currently has %d words" % self.size

    def get_word(self, index):
        return self.index2word[index] if index<self.size else 'UNK'

    def get_index(self, word, add=True):
        if add and word not in self.wordSet:
            self.word2index[word] = self.size+1 # [CHANGE] spare 0 for masking
            self.index2word[self.size+1] = word # [CHANGE] same
            self.wordSet.add(word)
            self.size += 1
        return self.word2index[word] if word in self.wordSet else self.word2index['UNK']

    def contains(self, word):
        return word in self.wordSet

    def add_sentence(self, sentence, returnIndices=True):
        indices = [self.get_index(word, add=True) for word in sentence.split()]
        return (indices,len(indices)) if returnIndices else None

    def add_document(self, docPath, returnIndices=True):
        with open(docPath, 'r') as doc:
            if returnIndices:
                indicesList, lengthList = [], []
                for line in doc:
                    indices,length = self.add_sentence(line,returnIndices)
                    if length<=0: continue # handle bad sentences in .txt.
                    indicesList.append(indices)
                    lengthList.append(length)
                return indicesList, lengthList
            else:
                for line in doc:
                    self.add_sentence(line,returnIndices=False)
                return None
    
    def to_words(self, indices):
        return [self.get_word(index) for index in indices]
    
    def to_sent(self, indices):
        return ' '.join(self.to_words(indices))
    
    def to_indices(self, words):
        return [self.get_index(word) for word in words]
    
class VocabControlIndexer(Indexer):
    
    def __init__(self, specialTokenList=None, wordSet=None):
        Indexer.__init__(self, specialTokenList=None)
        self.wordSet = wordSet
        specialTokenList = [] if specialTokenList is None else list(set(specialTokenList).union(set(['UNK'])))
        for i,word in enumerate(specialTokenList+list(wordSet)):
            self.word2index[word] = i
            self.index2word[i] = word
        self.wordSet.update(set(specialTokenList))
        self.size = len(specialTokenList) + len(wordSet)
    
    def get_index(self, word, add=True):
        if add and word not in self.wordSet:
            return self.word2index['UNK']
        return self.word2index[word]

class DataLoader:
    
    def __init__(self, dataDir):
        self.dataDir = dataDir # str
        self.dataDict = {path.split('.')[0]:self.dataDir+path
                         for path in os.listdir(self.dataDir) if path.endswith('.txt')}
        self.filenames = set(['train_source', 'train_target',
                              'test_source', 'test_target'])
        if self._filename_mismatch():
            raise Exception("Expected filenames under the directory:\n"+str(self.filenames)+
                            '\nGot:\n'+str(self.dataDict.keys())+'\n')
    
    def _filename_mismatch(self):
        return self.filenames - set(self.dataDict.keys()) != set([])


class DataIterator:
    """Data feeder by batch."""

    def __init__(self, indexer, pairs, lengths):
        """
        Args:
            indexer: an Indexer object.
            pairs: a list of pairs of token index lists.
            lengths: a list of pairs of sentence length lists.
        """
        self.indexer = indexer
        self.pairs = pairs
        self.lengths = lengths
        self.size = len(pairs)
        self.indices = range(self.size)

    def _get_padded_sentence(self, index, maxSentLen, maxTargetLen):
        """Pad a sentence pair by EOS (pad both to the largest length of respective batch).

        Args:
            index: index of a sentence & length pair in self.pairs, self.lengths.
            maxSentLen: the length of the longest source sentence.
            maxTargetLen: the length of the longest target sentence.
        Returns:
            padded source sentence (list), its length (int), 
            padded target sentence (list), its length (int).
        """
        sent1,sent2 = self.pairs[index][0], self.pairs[index][1]
        length1,length2 = self.lengths[index][0], self.lengths[index][1]
        paddedSent1 = sent1[:maxSentLen] if length1>maxSentLen else sent1+[self.indexer.get_index('EOS')]*(maxSentLen-length1)
        paddedSent2 = sent2[:maxTargetLen] if length2>maxTargetLen else sent2+[self.indexer.get_index('EOS')]*(maxTargetLen-length2)
        return paddedSent1,length1,paddedSent2,length2

    def random_batch(self, batchSize):
        """Random batching.

        Args:
            batchSize: size of a batch of sentence pairs and respective lengths.
        Returns:
            the batch of source sentence (Variable(torch.LongTensor())),
            the lengths of source sentences (numpy.array())
            and the same for target sentences and lengths.
        """
        batchIndices = np.random.choice(self.indices, size=batchSize, replace=False)
        batchSents,batchTargets,batchSentLens,batchTargetLens = [], [], [], []
        maxSentLen, maxTargetLen = np.array([self.lengths[index] for index in batchIndices]).max(axis=0)
        for index in batchIndices:
            paddedSent1,length1,paddedSent2,length2 = self._get_padded_sentence(index, maxSentLen, maxTargetLen)
            batchSents.append(paddedSent1)
            batchTargets.append(paddedSent2)
            batchSentLens.append(length1)
            batchTargetLens.append(length2)
        batchIndices = range(batchSize) # reindex from 0 for sorting.
        batchIndices = [i for i,l in sorted(zip(batchIndices,batchSentLens),key=lambda p:p[1],reverse=True)]
        batchSents = Variable(torch.LongTensor(np.array(batchSents)[batchIndices])).transpose(0,1) # <bc,mt> -> <mt,bc>
        batchTargets = Variable(torch.LongTensor(np.array(batchTargets)[batchIndices])).transpose(0,1)
        batchSentLens = np.array(batchSentLens)[batchIndices]
        batchTargetLens = np.array(batchTargetLens)[batchIndices]
        return batchSents, batchSentLens, batchTargets, batchTargetLens
    
class DataIteratorTransformer:
    """Data feeder by batch."""

    def __init__(self, indexer, pairs, lengths):
        """
        Args:
            indexer: an Indexer object.
            pairs: a list of pairs of token index lists.
            lengths: a list of pairs of sentence length lists.
        """
        self.indexer = indexer
        self.pairs = pairs
        self.lengths = lengths
        self.size = len(pairs)
        self.indices = range(self.size)

    def _get_padded_sentence(self, index, maxSentLen, maxTargetLen):
        """Pad a sentence pair by EOS (pad both to the largest length of respective batch).

        Args:
            index: index of a sentence & length pair in self.pairs, self.lengths.
            maxSentLen: the length of the longest source sentence.
            maxTargetLen: the length of the longest target sentence.
        Returns:
            padded source sentence (list), its length (int), 
            padded target sentence (list), its length (int).
        """
        sent1,sent2 = self.pairs[index][0], self.pairs[index][1]
        length1,length2 = self.lengths[index][0], self.lengths[index][1]
        paddedSent1 = sent1[:maxSentLen] if length1>maxSentLen else sent1+[0]*(maxSentLen-length1) # [CHANGE] 0 is padding token.
        paddedSent2 = sent2[:maxTargetLen] if length2>maxTargetLen else sent2+[0]*(maxTargetLen-length2)
        return paddedSent1,length1,paddedSent2,length2

    def random_batch(self, batchSize):
        """Random batching.

        Args:
            batchSize: size of a batch of sentence pairs and respective lengths.
        Returns:
            the batch of source sentence (Variable(torch.LongTensor())),
            the lengths of source sentences (numpy.array())
            and the same for target sentences and lengths.
        """
        batchIndices = np.random.choice(self.indices, size=batchSize, replace=False)
        batchSents,batchTargets,batchSentLens,batchTargetLens = [], [], [], []
        maxSentLen, maxTargetLen = np.array([self.lengths[index] for index in batchIndices]).max(axis=0)
        for index in batchIndices:
            paddedSent1,length1,paddedSent2,length2 = self._get_padded_sentence(index, maxSentLen, maxTargetLen)
            batchSents.append(paddedSent1)
            batchTargets.append(paddedSent2)
            batchSentLens.append(length1)
            batchTargetLens.append(length2)
        batchIndices = range(batchSize) # reindex from 0 for sorting.
        batchIndices = [i for i,l in sorted(zip(batchIndices,batchSentLens),key=lambda p:p[1],reverse=True)]
        batchSents = Variable(torch.LongTensor(np.array(batchSents)[batchIndices]))     # [CHANGE] <bc,mt> rather than <mt,bc>
        batchTargets = Variable(torch.LongTensor(np.array(batchTargets)[batchIndices])) # [CHANGE] same
        return batchSents, batchTargets

class BilingualDataIterator:
    """Data feeder by batch, for bilingual data."""

    def __init__(self, indexerLang1, indexerLang2, pairs, lengths, maxTargetLen=50):
        """
        Args:
            indexerLang1, indexerLang2: a pair of Indexer objects.
            pairs: a list of pairs of token index lists.
            lengths: a list of pairs of sentence length lists.
            maxTargetLen: uniform to decoding length later (for cross entropy computing).
        """
        self.indexerLang1 = indexerLang1
        self.indexerLang2 = indexerLang2
        self.pairs = pairs
        self.lengths = lengths
        self.maxTargetLen = maxTargetLen
        self.size = len(pairs)
        self.indices = range(self.size)

    def _get_padded_sentence(self, index, maxSentLen, maxTargetLen):
        """Pad a sentence pair by EOS (pad both to the largest length of respective batch).

        Args:
            index: index of a sentence & length pair in self.pairs, self.lengths.
            maxSentLen: the length of the longest source sentence.
            maxTargetLen: the length of the longest target sentence.
        Returns:
            padded source sentence (list), its length (int), 
            padded target sentence (list), its length (int).
        """
        sent1,sent2 = self.pairs[index][0], self.pairs[index][1]
        length1,length2 = self.lengths[index][0], self.lengths[index][1]
        paddedSent1 = sent1[:maxSentLen] if length1>maxSentLen else sent1+[self.indexerLang1.get_index('EOS')]*(maxSentLen-length1)
        paddedSent2 = sent2[:maxTargetLen] if length2>maxTargetLen else sent2+[self.indexerLang2.get_index('EOS')]*(maxTargetLen-length2)
        return paddedSent1,length1,paddedSent2,length2

    def random_batch(self, batchSize):
        """Random batching.

        Args:
            batchSize: size of a batch of sentence pairs and respective lengths.
        Returns:
            the batch of source sentence (Variable(torch.LongTensor())),
            the lengths of source sentences (numpy.array())
            and the same for target sentences and lengths.
        """
        batchIndices = np.random.choice(self.indices, size=batchSize, replace=False)
        batchSents,batchTargets,batchSentLens,batchTargetLens = [], [], [], []
        maxSentLen, _ = np.array([self.lengths[index] for index in batchIndices]).max(axis=0)
        for index in batchIndices:
            paddedSent1,length1,paddedSent2,length2 = self._get_padded_sentence(index, maxSentLen, self.maxTargetLen)
            batchSents.append(paddedSent1)
            batchTargets.append(paddedSent2)
            batchSentLens.append(length1)
            batchTargetLens.append(length2)
        batchIndices = range(batchSize) # reindex from 0 for sorting.
        batchIndices = [i for i,l in sorted(zip(batchIndices,batchSentLens),key=lambda p:p[1],reverse=True)]
        batchSents = torch.LongTensor(np.array(batchSents)[batchIndices]).transpose(0,1) # <bc,mt> -> <mt,bc>
        batchTargets = torch.LongTensor(np.array(batchTargets)[batchIndices]).transpose(0,1)
        batchSentLens = np.array(batchSentLens)[batchIndices]
        batchTargetLens = np.array(batchTargetLens)[batchIndices]
        return batchSents, batchSentLens, batchTargets, batchTargetLens
