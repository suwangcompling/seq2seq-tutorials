import os
import dill
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from nltk.translate.bleu_score import sentence_bleu as bleu
from nltk.translate.bleu_score import SmoothingFunction

from utils import *
from seq2seq_data_loader import *

SMOOTH = SmoothingFunction()

###############
## LOAD DATA ##
###############

loader = Seq2SeqDataLoader('mscoco/', freqCut=10000)
indexer, trainPairs, trainLens, testPairs, testLens = loader.load(specialTokenList=['EOS','SOS','PAD','UNK'])
%%time

glovePath = "/usr/local/google/home/wangsu/Documents/SEQ2SEQ/glove_embeddings/glove.6B.300d.txt"
embeddings = np.zeros((indexer.size,300)) 
word2embedding = {}
count = 0
print "Reading embeddings ..."
with open(glovePath, 'r') as f:
    for line in f:
        count += 1
        line = line.split()
        word = line[0]
        embedding = np.array(line[1:], dtype=float)
        word2embedding[word] = embedding
        if count%10000==0:
            print '... processed %d lines.' % count
print "\nLoading embeddings to matrix ..."
oovSize = 0
for i in range(indexer.size-4): # special tokens.
    word = indexer.get_word(i)
    if word in word2embedding:
        embeddings[i] = word2embedding[word]
    else:
        embeddings[i] = np.zeros(300)
        oovSize += 1
print "Done (#oov = %d)" % oovSize

###########
## MODEL ##
###########

class EncoderRNN(nn.Module):
    """Simple GRU encoder."""
    
    def __init__(self, inputSize, hiddenSize, embeddings=None, nLayers=2, dropout=0.1, bidirectional=True):
        """
        
        Args:
            inputSize: vocabulary size.
            hiddenSize: size of RNN hidden state.
            embeddings: pretrained, numpy.ndarray.
            nLayers: number of stacked layers.
            dropout: dropout rate.
            bidirectional: boolean.
        """
        # inputSize: vocabulary size.
        # hiddenSize: size for both embedding and GRU hidden.
        super(EncoderRNN, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.nLayers = nLayers
        self.embedding = nn.Embedding(inputSize, hiddenSize)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.dropoutLayer = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hiddenSize*2 if bidirectional else hiddenSize, hiddenSize)
        self.gru = nn.GRU(hiddenSize, hiddenSize, nLayers, dropout=dropout, bidirectional=bidirectional)
        
    
    def forward(self, inputs, inputsLen, hidden=None):
        """
        
        Args:
            inputs: Variable(torch.LongTensor()) of the shape <max-time,batch-size>.
            inputsLen: a list of input lengths with the shape <batch-size,>.
            hidden: input hidden state (initialized as None).
        """
        # inputs: <mt,bc>
        # inputsLen: <bc,> (a list).
        # hidden: <n_layer*n_direction,bc,h>
        
        
        embedded = self.embedding(inputs) # <mt,bc,h>
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, inputsLen)
            # 'packed' has a 'data' and a 'batch_sizes' field.
            #   'data' is a <sum(len),h> matrix (len is real lengths, not padded).
            #   'batch_sizes' has the number of non-zero batches at each time-step.
            # e.g. for this 'inputs'
            #    2     1     3     0     2
            #    6     8     1     6     2
            #    0     7     0     8     8
            #    6     4     2     1     1
            #    1     8     1     1     1
            #    6     1     1     1     1
            #    0     1     1     1     1
            #    1     1     1     1     1
            #    1     1     1     1     1
            #    1     1     1     1     1  
            # 'data' = 22 = 7+5+4+3+3 (1's are pads corresponding to 'EOS').
            # 'batch_sizes' = [5, 5, 5, 3, 2, 1, 1].
        outputs,hidden = self.gru(packed, hidden)#, dropout=dropout)
            # outputs: <sum(len),h*n_direction>.
            # hidden: <n_layer*n_direction,bc,h>
        hidden = torch.cat((hidden[:self.nLayers,:,:],hidden[self.nLayers:,:,:]),dim=-1)
            # two <n_layer,bc,h> (fw/bw) concate on last dim.
            # hidden: <n_layer,bc,2h>
        hidden = self.linear(hidden)
            # shrink it down to <n_layer,bc,h>, also get info for concat bidirections.
        outputs, outputsLen = torch.nn.utils.rnn.pad_packed_sequence(outputs)
            # outputs: <mt,bc,h*n_direction>
            # outputsLen: same as the 'batch_sizes' field of 'packed'. 
        outputs = outputs[:,:,:self.hiddenSize] + outputs[:,:,self.hiddenSize:]
            # add bidirectional outputs (for attention later)
            # outputs: <mt,bc,h>
        outputs = self.dropoutLayer(outputs)
        return outputs, hidden
    

class AttentionDecoderRNN(nn.Module):
    """Simple GRU decoder (Bahdanau attention)."""
    
    def __init__(self, hiddenSize, outputSize, embeddings=None, nLayers=2, dropout=0.1, residual=True):
        """
        
        Args:
            hiddenSize: GRU hidden state size.
            outputSize: vocabulary size.
            embeddings: pretrained, numpy.ndarray.
            nLayers: number of stacked layers.
            dropout: dropout rate.
            residual: boolean, whether establish residual link or not.
        """
        super(AttentionDecoderRNN, self).__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.nLayers = nLayers
        self.residual = residual
        self.embedding = nn.Embedding(outputSize, hiddenSize)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.dropoutLayer = nn.Dropout(p=dropout)
        self.gru = nn.GRU(2*hiddenSize, hiddenSize, nLayers) 
        self.out = nn.Linear(2*hiddenSize, outputSize)
            # inputSize doubles because concatted context of same hiddenSize.
        self.linear = nn.Linear(hiddenSize, hiddenSize)

    def forward(self, inputs, hidden, context, encoderOutput, encoderInputsLen):
        """
        
        Args:
            inputs: inputs to decoder, of the shape <batch-size,> (1 time-step).
            hidden: <n_layers*n_directions,batch-size,hidden-size>.
            context: context vector made using attention, <batch-size,hidden-size>.
            encoderOutput: <max-time,batch-size,hidden-size>.
            NB: all are Variable(torch.LongTensor()).
        Returns:
            output: <batch-size,vocab-size>.
            hidden: <n_layers*n_directions,batch-size,hidden-size>.
            context: <batch-size,hidden-size>.
            attentionWeights: <batch-size,max-time>.
        """
            # inputs: <bc,>
            # hidden: <n_layer*n_direction,bc,h>
            # context: <bc,h>
            # encoderOutput: <mt,bc,h>  
        batchSize = inputs.size(0)
        encoderOutputLen = encoderOutput.size(0)
        embedded = self.embedding(inputs).view(1,batchSize,self.hiddenSize) # <mt=1,bc,h>
        inputs = torch.cat((embedded,context.unsqueeze(0)),2)
            # unsqueeze: <bc,h> -> <mt=1,bc,h>
            # concat: <mt,bc,h> & <mt,bc,h> @2 -> <mt,bc,2h>
        output, hidden = self.gru(inputs, hidden)#, dropout=dropout)
            # IN: <mt=1,bc,2h>, <n_layer*n_direction,bc,h>
            # OUT: <mt=1,bc,h>, <n_layer*n_direction,bc,h>
        output = self.dropoutLayer(output)
        hidden = hidden + embedded if self.residual else hidden
        attentionWeights = Variable(torch.zeros(batchSize,encoderOutputLen)).cuda()
        for b in range(batchSize):
            rawAttentionWeight = torch.mm(self.linear(encoderOutput[:,b,:]), 
                                          hidden[:,b,:][-1].unsqueeze(1)).squeeze()
                # op1. linear transformation on encoderOutput (dot energy).
                # op2. select <mt,h> and <1,h> slices (from <mt,bc,h> and <1,bc,h>).
                # op3. sel hidden last dim <h,> and expand -> <mt,h> & <h,1> now.
                # op4. matmul -> <mt,1>.
                # op5. squeeze -> <mt,>
            if b>0:
                mask = Variable(torch.FloatTensor(np.array([1 if i<=encoderInputsLen[b] else 0 for i in range(encoderOutputLen)]))).cuda()
                    # encoder outputlen is the largest length in the current input batch.
                    # <= rather than <: including EOS
                rawAttentionWeight = F.softmax(rawAttentionWeight, dim=-1) * mask
                    # first softmax to get rid of negative numbers
            attentionWeights[b] = F.softmax(rawAttentionWeight, dim=-1)
                # normalize to get a distribution.
            # result: <bc,mt> attention matrix, normalized along mt.
        multiDiag = Variable(torch.eye(batchSize).expand(self.hiddenSize,batchSize,batchSize),
                             requires_grad=False).cuda()
            # op1. eye -> <bc,bc> diagonal matrix mask.
            # op2. expand -> <h,bc,bc>, same shape as attended encoderOutput.
            # op3. Variable/grad=false: same type as attended encoderOutput.
        context = (torch.matmul(attentionWeights, encoderOutput.permute(2,0,1)) * multiDiag).sum(dim=2).transpose(0,1)
            # op1. masking -> <h,bc,bc>, with the last 2 dims only have non-zero diag elems.
            # op2. compress 1 bc dimension (useless, because its diag).
            # op3. <h,bc> -> <bc,h>, keep input shape.
        output = output.squeeze(0)
            # output squeeze: <mt=1,bc=1,h> -> <bc,h>, to concat with context
        output = F.log_softmax(F.tanh(self.out(torch.cat((output,context),1))),dim=-1)
            # concat: <bc,h> & <bc,h> @1 -> <bc,2h>
            # linear->tahn/out: <bc,2h> * <2h,vocab> -> <bc,vocab>
            # softmax: along dim=-1, i.e. vocab.  
        return output, hidden, context, attentionWeights
            # full output for visualization:
            #   output: <bc,vocab>
            #   hidden: <n_layer*n_direction,bc,h>
            #   context: <bc,h>
            #   attentionWeights: <bc,mt> 

class Seq2Seq:
    """Encoder-Decoder model with Bahdanau attention, stacking and residual links."""
    
    def __init__(self, indexer, trainPairs, trainLens, testPairs, testLens,
                 embeddings=None,
                 batchSize=5, hiddenSize=10,
                 nLayers=2, dropout=0.1, residual=True, bidirectional=True, 
                 lr=1e-4, lrDecay=0.95, lrDecayFreq=100,
                 l2Reg=0.5,
                 enforcingRatio=0.5, clip=5.0,
                 maxDecodingLen=10,
                 resultSavePath='toy/results.txt'):
        """
        
        Args:
            indexer: an Indexer object.
            trainPairs, testPairs: each is a list of pairs of word index list.
            trainLens, testLens: each is a list of pairs of length of word index list.
            batchSize: int. (default=5)
            hiddenSize: int. (default=10)
            nLayers: number of GRU stacking layers. (default=2)
            dropout: dropout rate. (default=0.1)
            residual: boolean, whether to establish residual links. (default=True)
            bidirectional: boolean.
            lr: learning rate, float. (default=1e-4 with Adam)
            lrDecay: rate at which lr drops per m batches.
            lrDecayFreq: the number of batches per lr decay.
            l2Reg: ridge regression.
            enforcingRatio: the percentage of teacher-enforced training. (default=0.5)
            clip: gradient clip cap, float. (default=5.0)
            maxDecodingLen: max #tokens generated by decoder before stopping.
            resultSavePath: (input,prediction,target) sentence triples file path.
        """
        self.indexer = indexer
        self.trainIter = DataIterator(indexer, trainPairs, trainLens, maxTargetLen=maxDecodingLen)
        self.testIter = DataIterator(indexer, testPairs, testLens, maxTargetLen=maxDecodingLen)
        self.embeddings = embeddings
        self.batchSize = batchSize
        self.hiddenSize = hiddenSize
        self.nLayers = nLayers
        self.dropout = dropout
        self.residual = residual
        self.bidirectional = bidirectional
        self.lr = lr
        self.lrDecay = lrDecay
        self.lrDecayFreq = lrDecayFreq
        self.l2Reg = l2Reg
        self.enforcingRatio = enforcingRatio
        self.clip = clip
        self.maxDecodingLen = maxDecodingLen
        self.resultSavePath = resultSavePath
        self._build_model()
    
    def _build_model(self):
        """Specify computational graph."""
        
        
        self.encoder = EncoderRNN(self.indexer.size, self.hiddenSize, self.embeddings,
                                  nLayers=self.nLayers, dropout=self.dropout, bidirectional=self.bidirectional).cuda()
        self.decoder = AttentionDecoderRNN(self.hiddenSize, self.indexer.size, self.embeddings,
                                           nLayers=self.nLayers, dropout=self.dropout, residual=self.residual).cuda()
        self.encoderOptim = optim.Adam(self.encoder.parameters(), self.lr, weight_decay=self.l2Reg)
        self.decoderOptim = optim.Adam(self.decoder.parameters(), self.lr, weight_decay=self.l2Reg)
        self.criterion = nn.NLLLoss()
    
    def _model_config(self):
        return 'Vocab Size = ' + str(self.indexer.size) + '\n' + \
               'Train/Test Size = ' + str(self.trainIter.size)+'/'+str(self.testIter.size) + '\n' + \
               'batchSize = ' + str(self.batchSize) + '; hiddenSize = ' + str(self.hiddenSize) + '\n' + \
               'nLayers = ' + str(self.nLayers) + '; dropout = ' + str(self.dropout) + '\n' + \
               'residual = ' + str(self.residual) + '; learning rate = ' + str(self.lr) + '\n' + \
               'learning rate decay = ' + str(self.lrDecay) + ' per ' + str(self.lrDecayFreq) + ' batches/steps\n' + \
               'regularization (l2) = ' + str(self.l2Reg) + '\n' + \
               'teacher enforce ratio = ' + str(self.enforcingRatio) + '; clip = ' + str(self.clip) + '\nn'
    
    def _lr_decay(self, encOptim, decOptim):
        self.lr *= self.lrDecay
        for paramGroup in encOptim.param_groups:
            paramGroup['lr'] = self.lr
        for paramGroup in decOptim.param_groups:
            paramGroup['lr'] = self.lr
        return encOptim, decOptim
    
    def _train_step(self):
        """One step of training."""
        inputs, inputsLen, targets, targetsLen = self.trainIter.random_batch(self.batchSize)
        inputs, targets = inputs.cuda(), targets.cuda()
        self.encoderOptim.zero_grad()
        self.decoderOptim.zero_grad()
        loss = 0
        # Run encoder
        
#         print type(inputs)
#         assert 1==0
        
        encoderHidden = None
        encoderOutput, encoderHidden = self.encoder(inputs, inputsLen, encoderHidden)    
        # Run decoder
        decoderInput = Variable(torch.LongTensor([self.indexer.get_index('SOS')]*self.batchSize)).cuda()
        decoderContext = Variable(torch.zeros(self.batchSize,self.decoder.hiddenSize)).cuda()
        decoderHidden = encoderHidden
        enforce = random.random() < self.enforcingRatio
        currDecodingLen = min(max(targetsLen),self.maxDecodingLen)
        decoderOutputAll = Variable(torch.zeros(currDecodingLen,self.batchSize,self.decoder.outputSize)).cuda()
            # <mt-max,bc,vocab>
        mask = torch.LongTensor([1]*self.batchSize).cuda()
            # start with 1, a cell turns 0 to mask out generation after an EOS is seen.
        for di in range(currDecodingLen):
            decoderOutput,decoderHidden,decoderContext,attentionWeights = self.decoder(decoderInput,
                                                                                       decoderHidden,
                                                                                       decoderContext, 
                                                                                       encoderOutput,
                                                                                       inputsLen)
            decoderOutputAll[di] = decoderOutput
            if enforce:
                decoderInput = targets[di] # <== targets is <mt,bc>
            else:
                topValues,topIndices = decoderOutput.data.topk(1) # <bc,1>
                topIndices = topIndices.squeeze()# topIndices = <bc,>
                for b in range(self.batchSize):
                    if topIndices[b] == 0: # EOS
                        mask[b] = 0
                topIndices = topIndices * mask
                decoderInput = Variable(topIndices).cuda()     
        # Batch cross entropy
            # requires arg1/pred = <#entries,vocab>, arg2/target = <#entries,>
        decoderOutputAll = decoderOutputAll.view(-1, self.decoder.outputSize)
            # reshape to <mt*bc,vocab>
        targets = targets[:currDecodingLen,:].contiguous().view(-1)
            # reshape to <mt*bc>
        loss = self.criterion(decoderOutputAll, targets)
        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.clip)
        self.encoderOptim.step()
        self.decoderOptim.step()
        return loss.data[0] / targetsLen.sum()
    
    def train(self, nEpochs=1, epochSize=100, printEvery=5):
        """Train on loaded data upon construction.
        
        Args:
            nEpochs: number of epochs.
            epochSize: number of batches trained in an epoch.
            printEvery: frequency of results report.
        """
        globalStep = 0
        averageLoss = 0
        start = time.time()
        for e in range(nEpochs):
            epochLoss = 0
            for step in range(epochSize):
                loss = self._train_step()
                if step!=0 and step%printEvery==0:
                    print("Step %d average loss = %.4f (time: %.2f)" % (step, loss, # batch mean.
                                                                        time.time()-start))
                    start = time.time()
                epochLoss += loss
                globalStep += 1
                if globalStep%self.lrDecayFreq==0:
                    print '  [LR DECAY] from %.10f to %.10f at step %d' % (self.lr, self.lr*self.lrDecay, globalStep)
                    self.encoderOptim, self.decoderOptim = self._lr_decay(self.encoderOptim, self.decoderOptim)
            epochLoss /= epochSize
            averageLoss += epochLoss
            print("\nEpoch %d loss = %.4f\n" % (e+1,epochLoss))
            averageBleu = self.evaluate_random(size=self.batchSize, saveResults=False, printResults=True)
        averageLoss /= nEpochs
        print("\nGrand average loss = %.4f\n" % averageLoss) 
        
    def _clear_special_tokens(self, words):
        """Clear all the PAD, UNK, EOS to avoid inflated BLEU.
        
        Args:
            words: a list of tokens.
        Returns:
            a list of tokens which are not special tokens.
        """
        return [word for word in words if word not in set(["PAD","UNK","EOS","SOS"])]

    def evaluate_pair(self, predWords, targetWords):
        """Compute the BLEU score of a prediction given a reference.
        
        Args:
            predWords: predicted words (a list of strings).
            targetWords: reference, same type as preWords.
        Returns:
            The BLEU score (uses = nltk.translate.bleu_score.sentence_bleu).
        """
        return bleu([self._clear_special_tokens(targetWords)], 
                     self._clear_special_tokens(predWords), smoothing_function=SMOOTH.method3)

    def _print_heatmap(self, attentionWeightsAll, inputs, predictions, targets):
        """Plot alignment heatmaps.
        
        Args:
            attentionWeightsAll: shape <bc,mt-dec=maxDecodingLen,mt-enc=inputsLen>
            inputs: <bc,mt-enc=inputsLen>
            predictions: <bc,mt-dec=maxDecodingLen>
            targets: <bc,mt-dec=targetsLen>
            NB: all types = numpy.ndarray.
        """

        print '=========== HEATMAPS =========='
        for aw,iw,pw,tw in zip(attentionWeightsAll,inputs,predictions,targets): # <mt,vocab>
            _, ax = plt.subplots()
            xticks = self.indexer.to_words(iw)
            yticks = [p+'-'+t for p,t in zip(self.indexer.to_words(pw),self.indexer.to_words(tw))]
            ax = sns.heatmap(aw, xticklabels=xticks, yticklabels=yticks)
            ax.xaxis.tick_top()
            plt.show()
        print '===============================\n\n'
        
    def evaluate_random(self, size, saveResults, printResults=True, endOfEpochPlot=True):
        """Randomly evaluate samples from the test set (which is loaded upon construction).
        
        Args:
            size: number of samples evaluated (as a single batch).
            printResults: print input, prediction and gold translation to console. (default=True)
        Returns:
            The average BLEU score in the batch.
        """
        self.encoder.eval()
        self.decoder.eval()
        inputs, inputsLen, targets, targetsLen = self.testIter.random_batch(size)
        inputs, targets = inputs.cuda(), targets.cuda()
        # Run encoder
        encoderHidden = None
        encoderOutput, encoderHidden = self.encoder(inputs, inputsLen, encoderHidden)
        # Run decoder
        decoderInput = Variable(torch.LongTensor([self.indexer.get_index('SOS')]*size)).cuda()
        decoderContext = Variable(torch.zeros(size,self.decoder.hiddenSize)).cuda()
        decoderHidden = encoderHidden
        predictions = []
        attentionWeightsAll = []
        for di in range(self.maxDecodingLen):
            decoderOutput,decoderHidden,decoderContext,attentionWeights = self.decoder(decoderInput,
                                                                                       decoderHidden,
                                                                                       decoderContext, 
                                                                                       encoderOutput,
                                                                                       inputsLen)
            attentionWeightsAll.append(attentionWeights)
            topValues,topIndices = decoderOutput.data.topk(1) # <bc,1>
            decoderInput = Variable(topIndices.squeeze()).cuda() # <bc,1> -> <bc,>
            predictions.append(topIndices.view(-1).cpu().numpy())
        attentionWeightsAll = np.array([aw.data.cpu().numpy() for aw in attentionWeightsAll])
        attentionWeightsAll = attentionWeightsAll.swapaxes(0,1) # <mt-dec,bc,mt-enc> -> <bc,mt-dec,mt-enc> 
        inputs = inputs.data.cpu().numpy().transpose()
        predictions = np.array(predictions).transpose() # <mt,bc> -> <bc,mt>
        targets = targets.data.cpu().numpy().transpose()
        
        if endOfEpochPlot:
            self._print_heatmap(attentionWeightsAll, inputs, predictions, targets)
        bleuList = []
        results = []
        for i,(input,pred,target) in enumerate(zip(inputs,predictions,targets)):
            inputWords = self._clear_special_tokens(self.indexer.to_words(input))
            predWords = self._clear_special_tokens(self.indexer.to_words(pred))
            targetWords = self._clear_special_tokens(self.indexer.to_words(target))
            bleuCurr = self.evaluate_pair(predWords, targetWords)
            bleuList.append(bleuCurr)
            inputSent = ' '.join(inputWords)
            predSent = ' '.join(predWords)
            targetSent = ' '.join(targetWords)
            results.append([inputSent, predSent, targetSent])
            if printResults:
                print("Example %d" % (i+1))
                print("INPUT >> %s" % inputSent)
                print("PRED >> %s" % predSent)
                print("TRUE >> %s" % targetSent)
                print("[BLEU] %.2f\n" % bleuCurr)
        averageBleu = np.mean(bleuList)
        if saveResults:
            return averageBleu, results
        return averageBleu

    def evaluate(self, nBatches=10, saveResults=True):
        """Randomly evaluate a given number of batches.
        
        Args:
            nBatches: the number of random batches to be evaluated.
        """
        averageBleuList = []
        for i in range(nBatches):
            if saveResults:
                averageBleu, results = self.evaluate_random(self.batchSize, saveResults, printResults=False)
                averageBleuList.append(averageBleu)
                with open(self.resultSavePath, 'a') as f:
                    if i==0:
                        f.write(self._model_config())
                        f.write('=================================\n')
                    for input,pred,target in results:
                        f.write('INPUT  >> ' + input + '\n')
                        f.write('PRED   >> ' + pred + '\n')
                        f.write('TARGET >> ' + target + '\n\n')
            else:
                averageBleuList.append(self.evaluate_random(self.batchSize, saveResults, printResults=False))
        message = "Average BLEU score over %d examples is %.4f" % (self.batchSize*nBatches, 
                                                                   np.mean(averageBleuList))
        with open(self.resultSavePath, 'a') as f:
            f.write('=================================\n')
            f.write(message)
        print message
            
    def evaluate_given(self, sent, maxLen=20):
        """Evaluate a give sentence.
        
        Args:
            sentence: a single string. OOVs are treated as UNKs.
            maxLen: the max number of decoding steps.
        """
        sent = sent.split()
        sentCode = [self.indexer.get_index(word,add=False) for word in sent]
        if any(i==-1 for i in sentCode):
            raise Exception("This sentence contains out of vocabulary words!")
        input = Variable(torch.LongTensor(sentCode)).cuda().view(-1,1)
        inputLen = np.array([len(sentCode)])
        # Run encoder
        encoderHidden = None
        encoderOutput, encoderHidden = self.encoder(input, inputLen, encoderHidden)
        # Run decoder
        decoderInput = Variable(torch.LongTensor([self.indexer.get_index('SOS')]*1)).cuda()
        decoderContext = Variable(torch.zeros(1,self.decoder.hiddenSize)).cuda()
        decoderHidden = encoderHidden
        pred = []
        for di in range(maxLen):
            decoderOutput,decoderHidden,decoderContext,attentionWeights = self.decoder(decoderInput,
                                                                                       decoderHidden,
                                                                                       decoderContext, 
                                                                                       encoderOutput)
            topValues,topIndices = decoderOutput.data.topk(1) # <bc,1>
            decoderInput = Variable(topIndices.squeeze()).cuda() # <bc,1> -> <bc,>
            predIndex = topIndices.view(-1).cpu().numpy()[0]
            if predIndex == self.indexer.get_index('EOS'):
                break
            pred.append(predIndex)
        print("INPUT >> %s" % ' '.join(sent))
        print("PRED >> %s\n" % ' '.join(self.indexer.to_words(pred))) 
       
#########
## RUN ##      
#########

%%time

s2s = Seq2Seq(indexer, trainPairs, trainLens, testPairs, testLens,
              embeddings=embeddings,
              batchSize=32, hiddenSize=300, bidirectional=True,
              nLayers=2, dropout=0.3, residual=True, 
              lr=1e-3, lrDecay=0.95, lrDecayFreq=50000000,
              l2Reg=0.001,
              enforcingRatio=0.5, clip=20.0,
              maxDecodingLen=10,
              resultSavePath='mscoco/resultargetsLents.txt')
s2s.train(nEpochs=10, epochSize=1000, printEvery=50)
s2s.evaluate(nBatches=10, saveResults=True)
torch.save(s2s, 'mscoco/seq2seq.ckpt')
