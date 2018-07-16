# 2018 Su Wang (wangsu@google.com)

import os
import dill
import argparse
from utils import Indexer, DataLoader

class ParaphraseDataLoad(DataLoader):
    
    def __init__(self, dataDir):
        DataLoader.__init__(self, dataDir)
    
    def load(self, specialTokenList=None):
        indexer = Indexer(specialTokenList)
        print "... loading training data."
        trainPairs,trainLens = self._load_pairs(indexer,
                                                self.dataDict['train_source'],
                                                self.dataDict['train_target'])
        print "... loading test data."
        testPairs,testLens = self._load_pairs(indexer,
                                              self.dataDict['test_source'],
                                              self.dataDict['test_source'])
        print "Done!\n"
        return indexer,trainPairs,trainLens,testPairs,testLens

    def _load_pairs(self, indexer, sourcePath, targetPath):
        sourceSents,sourceLens = indexer.add_document(sourcePath,returnIndices=True)
        targetSents,targetLens = indexer.add_document(targetPath,returnIndices=True)
        return zip(sourceSents, targetSents), zip(sourceLens, targetLens)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help="Directory to data .txt files.",
                        type=str, default="EMPTY")
    parser.add_argument('--save_path', help="Path to saved dill .p file.",
                        type=str, default=".default_dill.p")
    args = parser.parse_args()
    if args.data_dir=="EMPTY":
        raise Exception("Please enter a valid directory path, under which you data should be:\n"+
                        "  train_source.txt | train_target.txt | test_source.txt | test_target.txt\n"+
                        "and no other .txt file\n"+
                        "Each file should have each line as a whitespace-separated sentence.")
    loader = ParaphraseDataLoad(args.data_dir)
    print "Loading data ..."
    indexer,trainPairs,trainLens,testPairs,testLens = loader.load(specialTokenList=["PAD","UNK","SOS","EOS"])
    print "\nTrain Size = " + str(len(trainPairs)) + \
          "; Test Size = " + str(len(testPairs)) + \
          "\nVocab Size = " + str(indexer.size) + "\n"
    print "Saving data ..."
    dill.dump((indexer,trainPairs,trainLens,testPairs,testLens),open(args.save_path,'wb'))
    print("Done. Data saved as %s\n" % args.save_path)
