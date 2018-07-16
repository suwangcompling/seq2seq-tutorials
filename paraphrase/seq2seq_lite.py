import os
import dill
import random
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from nltk.translate.bleu_score import sentence_bleu as bleu

from utils import Indexer
from utils import DataIterator

class EncoderRNN(nn.Module):
    """Simple GRU encoder."""
    
    def __init__(self, inputSize, hiddenSize, nLayers=2, dropout=0.1):
        """
        
        Args:
            inputSize: vocabulary size.
            hiddenSize: size of RNN hidden state.
            nLayers: number of stacked layers.
            dropout: dropout rate.
        """
        # inputSize: vocabulary size.
        # hiddenSize: size for both embedding and GRU hidden.
        super(EncoderRNN, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.nLayers = nLayers
        self.dropout = dropout
        self.embedding = nn.Embedding(inputSize, hiddenSize)
        self.gru = nn.GRU(hiddenSize, hiddenSize, nLayers, dropout)
    
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
        outputs,hidden = self.gru(packed, hidden)
            # outputs: same format as 'packed'.
            # hidden: <n_layer*n_direction,bc,h>
        outputs, outputsLen = torch.nn.utils.rnn.pad_packed_sequence(outputs)
            # outputs: <mt,bc,h>
            # outputsLen: same as the 'batch_sizes' field of 'packed'.   
        return outputs, hidden
    
class LinearAttention(nn.Module):
    """Basic linear attention layer."""
    
    def __init__(self, hiddenSize):
        """
        
        Args:
            hiddenSize: size of RNN hidden state.
        """
        super(LinearAttention, self).__init__()
        self.hiddenSize = hiddenSize
        self.attention = nn.Linear(hiddenSize, hiddenSize)
    
    def forward(self, hidden, encoderOutput):
        """
        
        Args:
            hidden: previous decoder hidden state of the shape <max-time=1,batch-size,hidden-size>.
            encoderOutput: encoder's entire outputs (for attention), with the shape <max-time,batch-size,hidden-size>.
        Returns:
            attention weights, of the shape <batch-size,1,max-time> (the singleton dimension is for convenience of processing).
        """
        # hidden: <1,bc,h>
        # encoderOutput: <mt,bc,h>
        encoderOutputLen, batchSize = encoderOutput.size(0), encoderOutput.size(1)
        encoderOutputLen = len(encoderOutput)
        attentionEnergies = Variable(torch.zeros(batchSize, encoderOutputLen)) # <bc,mt>
        for b in range(batchSize):
            for i in range(encoderOutputLen):
                attentionEnergies[b,i] = self.score(hidden[:,b],encoderOutput[i,b].unsqueeze(0))
                    # hidden[:,b] selects a <1,h> from <1,bc,h>
                    # encoderOutput[i,b] selects a <h,> from <mt,bc,h>
                    #   then unsqueeze(0) to add a first dimension to make <1,h>
                    # score thus takes <1,h> and <1,h>
        return F.softmax(attentionEnergies, dim=-1).unsqueeze(1)
            # first softmax along the mt dimension of <bc,mt>,
            # then unsqueeze(1) to make <bc,1,mt>, technical convenience.
        
    def score(self, hidden, encoderOutput):
        """
        
        Args: 
            hidden: a slice of hidden state, of the shape <1,hidden-size>.
            encoderOutput: a slice of encoder outputs, of the shape <1,hidden-size>.
        Returns:
            a slice of attention weights: of the shape <1,hidden-size>
        """
            # hidden: <bc=1,h>
            # encoderOutput: <bc=1,h> (1 time step).
        energy = self.attention(encoderOutput)
            # linear attention: <bc,h> * <h,h> -> <bc,h>   
        energy = hidden.dot(energy)
            # dot: <bc,h> * <bc,h> -> <bc,h>
            # .dot smartly find fitting dimensions.
        return energy

class LuongDecoderRNN(nn.Module):
    """Luong attention."""
    
    def __init__(self, hiddenSize, outputSize, nLayers=2, dropout=0.1, residual=True):
        """
        
        Args:
            hiddenSize: GRU hidden state size.
            outputSize: vocabulary size.
            nLayers: number of stacked layers.
            dropout: dropout rate.
            residual: boolean, whether establish residual link or not.
        """
        super(LuongDecoderRNN, self).__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.nLayers = nLayers
        self.dropout = dropout
        self.residual = residual
        self.embedding = nn.Embedding(outputSize, hiddenSize)
        self.gru = nn.GRU(2*hiddenSize, hiddenSize, nLayers, dropout) 
        self.out = nn.Linear(2*hiddenSize, outputSize)
            # inputSize doubles because concatted context of same hiddenSize.
        self.linearAttention = LinearAttention(hiddenSize)
        
    def forward(self, inputs, hidden, context, encoderOutput):
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
            attentionWeights: <batch-size,1,max-time>.
        """
            # inputs: <bc,>
            # hidden: <n_layer*n_direction,bc,h>
            # context: <bc,h>
            # encoderOutput: <mt,bc,h>  
        batchSize = inputs.size(0)
        embedded = self.embedding(inputs).view(1,batchSize,self.hiddenSize) # <mt=1,bc,h>
        inputs = torch.cat((embedded,context.unsqueeze(0)),2)
            # unsqueeze: <bc,h> -> <mt=1,bc,h>
            # concat: <mt,bc,h> & <mt,bc,h> @2 -> <mt,bc,2h>
        output, hidden = self.gru(inputs, hidden)
            # IN: <mt=1,bc,2h>, <n_layer*n_direction,bc,h>
            # OUT: <mt=1,bc,h>, <n_layer*n_direction,bc,h> 
        hidden = hidden+embedded if self.residual else hidden
            # broachcast addition: <n_layer*n_direction,bc,h> + <mt=1,bc,h>
            #   = <n_layer*n_direction,bc,h>.
        attentionWeights = self.linearAttention(output,
                                                encoderOutput)
            # squeeze: <mt=1,bc,h> -> <bc,h>
            # attentionWeights: <bc=1,1,mt>
        context = attentionWeights.bmm(encoderOutput.transpose(0,1))
            # transpose: <mt,bc,h> -> <bc,mt,h>
            # bmm (batched matrix multiplication): 
            #   <bc,1,mt> & <bc,mt,h> -> <bc,1,h>
        output = output.squeeze(0)
        context = context.squeeze(1)
            # output squeeze: <mt=1,bc=1,h> -> <bc,h>
            # context squeeze: <bc=1,1,h> -> <bc,h>
        output = F.log_softmax(F.tanh(self.out(torch.cat((output,context),1))),dim=-1)
            # concat: <bc,h> & <bc,h> @1 -> <bc,2h>
            # linear->tahn/out: <bc,2h> * <2h,vocab> -> <bc,vocab>
            # softmax: along dim=-1, i.e. vocab.
        return output, hidden, context, attentionWeights
            # full output for visualization:
            #   output: <bc,vocab>
            #   hidden: <n_layer*n_direction,bc,h>
            #   context: <bc,h>
            #   attentionWeights: <bc,1,mt>
            
def batch_cross_entropy(decoderOutputAll, targets, targetsLen, batchSize):
    """Compute cross entropy score for a batch of decoder outputs.
    
    Args:
        decoderOutputAll: <batch-size,max-time,vocab-size>, type Variable(torch.LongTensor()).
        targets: <batch-size,max-time>, type Variable(torch.LongTensor()).
        targetsLen: <batch-size,>, type list.
    Returns:
        average loss, type Variable(torch.LongTensor()).
    """
    # decoderOutputAll: <bc,mt,vocab> (transposed in train function).
    # targets: <bc,mt>
    # targetsLen: <bc,> (a list).
    logitsFlat = decoderOutputAll.view(-1, decoderOutputAll.size(-1))
        # <bc,mt,vocab> -> <bc*mt,vocab>
    logProbsFlat = F.log_softmax(logitsFlat,dim=-1)
        # <bc,mt,vocab>, with dim vocab has log probs.
    targetsFlat = targets.view(-1,1)
        # <bc,mt> -> <bc*mt,1>
    lossesFlat = -torch.gather(logProbsFlat,dim=1,index=targetsFlat)
        # <bc,mt,vocab> -> <bc*mt,1>
    losses = lossesFlat.view(*targets.size())
        # reshape: <bc*mt,1> -> <bc,mt>
    # Make a mask
    #   requires: lengths, maxLen
    maxLen = max(targetsLen)
    seqRange = torch.arange(maxLen).long()
        # generate a maxLen tensor of long type, <max-len,>
    seqRangeExpand = Variable(seqRange.unsqueeze(0).expand(batchSize,maxLen))
        # unsqueeze: <1,max-len>
        # expand: copy BATCH_SIZE times along first dimension
        #         second dim won't change as they are of the same length.
        #   e.g. for expand:
        #     >>> x = torch.Tensor([[1], [2], [3]])
        #     >>> x.size()
        #     torch.Size([3, 1])
        #     >>> x.expand(3, 4)
        #      1  1  1  1
        #      2  2  2  2
        #      3  3  3  3
        #     [torch.FloatTensor of size 3x4]
        #   finally we got <bc,max-len>.
    seqLenExpand = (Variable(torch.LongTensor(targetsLen)).unsqueeze(1).expand_as(seqRangeExpand))
        # unsqueeze: <bc,> -> <bc,1>
        # expand_as: <bc,1> -> <bc,max-len> 
    mask = seqRangeExpand < seqLenExpand
        # e.g. batch=2 case:
        #   seqRangeExpand is
        #     0 1 2 3
        #     0 1 2 3
        #   seqLenExpand is
        #     3 3 3 3 <= length of this sentence is 3
        #     2 2 2 2
        #   then we got a matrix that's elementwise results from the comparison.
        #     1 1 1 0
        #     1 1 0 0
        #   which means an elem=1 if it doesn't correspond to a padder.
    # Compute final loss
    losses = losses * mask.float() # zeroify all 0 elem in the mask.
    loss = losses.sum() / sum(targetsLen)
    return loss

class Seq2Seq:
    """Encoder-Decoder model with Luong attention, stacking and residual links."""
    
    def __init__(self, indexer, trainPairs, trainLens, testPairs, testLens, 
                 batchSize=5, hiddenSize=10,
                 nLayers=2, dropout=0.1, residual=True, 
                 lr=1e-4, enforcingRatio=0.5, clip=5.0,
                 resultSavePath='mscoco/results.txt'):
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
            lr: learning rate, float. (default=1e-4 with Adam)
            enforcingRatio: the percentage of teacher-enforced training. (default=0.5)
            clip: gradient clip cap, float. (default=5.0)
            resultSavePath: (input,prediction,target) sentence triples file path.
        """
        self.indexer = indexer
        self.trainIter = DataIterator(indexer, trainPairs, trainLens)
        self.testIter = DataIterator(indexer, testPairs, testLens)
        self.batchSize = batchSize
        self.hiddenSize = hiddenSize
        self.nLayers = nLayers
        self.dropout = dropout
        self.residual = residual
        self.lr = lr
        self.enforcingRatio = enforcingRatio
        self.clip = clip
        self.resultSavePath = resultSavePath
        self._build_model()
    
    def _build_model(self):
        """Specify computational graph."""
        self.encoder = EncoderRNN(self.indexer.size, self.hiddenSize, 
                                  nLayers=self.nLayers, dropout=self.dropout)
        self.decoder = LuongDecoderRNN(self.hiddenSize, self.indexer.size,
                                       nLayers=self.nLayers, dropout=self.dropout, residual=self.residual)
        self.encoderOptim = optim.Adam(self.encoder.parameters(), self.lr)
        self.decoderOptim = optim.Adam(self.decoder.parameters(), self.lr)
    
    def _model_config(self):
        return 'Vocab Size = ' + str(self.indexer.size) + '\n' + \
               'Train/Test Size = ' + str(self.trainIter.size)+'/'+str(self.testIter.size) + '\n' + \
               'batchSize = ' + str(self.batchSize) + '; hiddenSize = ' + str(self.hiddenSize) + '\n' + \
               'nLayers = ' + str(self.nLayers) + '; dropout = ' + str(self.dropout) + '\n' + \
               'residual = ' + str(self.residual) + '; learning rate = ' + str(self.lr) + '\n' + \
               'teacher enforce ratio = ' + str(self.enforcingRatio) + '; clip = ' + str(self.clip) + '\nn'
    
    def _train_step(self):
        """One step of training."""
        inputs, inputsLen, targets, targetsLen = self.trainIter.random_batch(self.batchSize)
        self.encoderOptim.zero_grad()
        self.decoderOptim.zero_grad()
        loss = 0
        # Run encoder
        encoderHidden = None
        encoderOutput, encoderHidden = self.encoder(inputs, inputsLen, encoderHidden)    
        # Run decoder
        decoderInput = Variable(torch.LongTensor([self.indexer.get_index('SOS')]*self.batchSize))
        decoderContext = Variable(torch.zeros(self.batchSize,self.decoder.hiddenSize))
        decoderHidden = encoderHidden
        enforce = random.random() < self.enforcingRatio
        maxTargetLen = max(targetsLen)
        decoderOutputAll = Variable(torch.zeros(maxTargetLen,self.batchSize,self.decoder.outputSize))
            # <mt-max,bc,vocab>
        for di in range(maxTargetLen):
            decoderOutput,decoderHidden,decoderContext,attentionWeights = self.decoder(decoderInput,
                                                                                       decoderHidden,
                                                                                       decoderContext, 
                                                                                       encoderOutput)
            decoderOutputAll[di] = decoderOutput
            if enforce:
                decoderInput = targets[di] # <== targets is <mt,bc>
            else:
                topValues,topIndices = decoderOutput.data.topk(1) # <bc,1>
                decoderInput = Variable(topIndices.squeeze()) # <bc,1> -> <bc,>
        # Sequence cross entropy
        loss = batch_cross_entropy(decoderOutputAll.transpose(0,1).contiguous(), 
                                   targets.transpose(0,1).contiguous(), 
                                   targetsLen,
                                   self.batchSize)
        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.clip)
        self.encoderOptim.step()
        self.decoderOptim.step()
        return loss.data[0] / targetsLen        
    
    def train(self, nEpochs=1, epochSize=100, printEvery=5):
        """Train on loaded data upon construction.
        
        Args:
            nEpochs: number of epochs.
            epochSize: number of batches trained in an epoch.
            printEvery: frequency of results report.
        """
        averageLoss = 0
        start = time.time()
        for e in range(nEpochs):
            epochLoss = 0
            for step in range(epochSize):
                loss = self._train_step()
                if step!=0 and step%printEvery==0:
                    print("Step %d average loss = %.4f (time: %.2f)" % (step, loss.mean(), # batch mean.
                                                                        time.time()-start))
                    start = time.time()
                epochLoss += loss.mean()
            epochLoss /= epochSize
            averageLoss += epochLoss
            print("\nEpoch %d loss = %.4f\n" % (e+1,epochLoss))
        averageLoss /= nEpochs
        print("\nGrand average loss = %.4f\n" % averageLoss) 
    
    def _clear_special_tokens(self, words):
        """Clear all the PAD, UNK, SOS, EOS to avoid inflated BLEU.
        
        Args:
            words: a list of tokens.
        Returns:
            a list of tokens which are not special tokens.
        """
        return [word for word in words if word not in set(["PAD","UNK","SOS","EOS"])]

    def evaluate_pair(self, predWords, targetWords):
        """Compute the BLEU score of a prediction given a reference.
        
        Args:
            predWords: predicted words (a list of strings).
            targetWords: reference, same type as preWords.
        Returns:
            The BLEU score (uses = nltk.translate.bleu_score.sentence_bleu).
        """
        return bleu([self._clear_special_tokens(targetWords)], 
                     self._clear_special_tokens(predWords))
        
    def evaluate_random(self, size, saveResults, printResults=True):
        """Randomly evaluate samples from the test set (which is loaded upon construction).
        
        Args:
            size: number of samples evaluated (as a single batch).
            printResults: print input, prediction and gold translation to console. (default=True)
        Returns:
            The average BLEU score in the batch.
        """
        inputs, inputsLen, targets, targetsLen = self.testIter.random_batch(size)
        # Run encoder
        encoderHidden = None
        encoderOutput, encoderHidden = self.encoder(inputs, inputsLen, encoderHidden)
        # Run decoder
        decoderInput = Variable(torch.LongTensor([self.indexer.get_index('SOS')]*size))
        decoderContext = Variable(torch.zeros(size,self.decoder.hiddenSize))
        decoderHidden = encoderHidden
        maxTargetLen = max(targetsLen)
        predictions = []
        for di in range(maxTargetLen):
            decoderOutput,decoderHidden,decoderContext,attentionWeights = self.decoder(decoderInput,
                                                                                       decoderHidden,
                                                                                       decoderContext, 
                                                                                       encoderOutput)
            topValues,topIndices = decoderOutput.data.topk(1) # <bc,1>
            decoderInput = Variable(topIndices.squeeze()) # <bc,1> -> <bc,>
            predictions.append(topIndices.view(-1).numpy())
        inputs = inputs.data.numpy().transpose()
        predictions = np.array(predictions).transpose() # <mt,bc> -> <bc,mt>
        targets = targets.data.numpy().transpose()
        bleuList = []
        results = []
        for i,(input,pred,target) in enumerate(zip(inputs,predictions,targets)):
            predWords = self.indexer.to_words(pred)
            targetWords = self.indexer.to_words(target)
            bleuCurr = self.evaluate_pair(predWords, targetWords)
            bleuList.append(bleuCurr)
            inputSent = self.indexer.to_sent(input)
            predSent = self.indexer.to_sent(pred)
            targetSent = self.indexer.to_sent(target)
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
                    for input,pred,target in results:
                        f.write('INPUT  >> ' + input + '\n')
                        f.write('PRED   >> ' + pred + '\n')
                        f.write('TARGET >> ' + target + '\n\n')
            else:
                averageBleuList.append(self.evaluate_random(self.batchSize, saveResults, printResults=False))
        print("Average BLEU score over %d examples is %.4f" % (self.batchSize*nBatches, 
                                                               np.mean(averageBleuList)))
            

            
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
        input = Variable(torch.LongTensor(sentCode)).view(-1,1)
        inputLen = np.array([len(sentCode)])
        # Run encoder
        encoderHidden = None
        encoderOutput, encoderHidden = self.encoder(input, inputLen, encoderHidden)
        # Run decoder
        decoderInput = Variable(torch.LongTensor([self.indexer.get_index('SOS')]*1))
        decoderContext = Variable(torch.zeros(1,self.decoder.hiddenSize))
        decoderHidden = encoderHidden
        pred = []
        for di in range(maxLen):
            decoderOutput,decoderHidden,decoderContext,attentionWeights = self.decoder(decoderInput,
                                                                                       decoderHidden,
                                                                                       decoderContext, 
                                                                                       encoderOutput)
            topValues,topIndices = decoderOutput.data.topk(1) # <bc,1>
            decoderInput = Variable(topIndices.squeeze()) # <bc,1> -> <bc,>
            predIndex = topIndices.view(-1).numpy()[0]
            if predIndex == self.indexer.get_index('EOS'):
                break
            pred.append(predIndex)
        print("INPUT >> %s" % ' '.join(sent))
        print("PRED >> %s\n" % ' '.join(self.indexer.to_words(pred))) 
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help="Path to formatted input to Seq2Seq. See Seq2Seq documentation.",
                        type=str,
                        default='mscoco/mscoco_formatted.p')
    parser.add_argument('--model_save_path', help="Path to saved Seq2Seq model.",
                        type=str,
                        default='mscoco/seq2seq.ckpt')
    parser.add_argument('--result_save_path', help="Path to save results.",
                        type=str, default='mscoco/results.txt')
    parser.add_argument('--clear_prev_result', help="Delete previously output results.",
                        type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=10)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--residual', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--enforce_ratio', type=float, default=0.5)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--epoch_size', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=5)
    parser.add_argument('--n_eval_batches', type=int, default=10)
    parser.add_argument('--max_decoding_length', type=int, default=20)
    args = parser.parse_args()
    
    option = raw_input("OPTIONS: r(etrain)/c(ontinue).\nr: Retrain new model\nc: Continue training of saved model.\n")
    if option=='r':
        if not os.path.exists(args.data_path):
            dataBuildMsg = """
                Data does not exist. Please make it in the following format:\n
                indexer: Indexer object which has files loaded (each is a file with space-separated words as lines).
                trainPairs, testPairs: each is a list of pairs of word index list.
                trainLens, testLens: each is a list of pairs of length of word index list.\n
                The order is indexer, trainPairs, trainLens, testPairs, testLens. Pickle it with dill.\n
            """
            raise Exception(dataBuildMsg)
        else:
            if os.path.exists(args.model_save_path):
                option = raw_input("Model exists. Hit c(ontinue) to overwrite it, (q)uit to quit.\n")
                if option=='q':
                    exit(0)
            indexer, trainPairs, trainLens, testPairs, testLens = dill.load(open(args.data_path, 'rb'))
            model = Seq2Seq(indexer, trainPairs, trainLens, testPairs, testLens,
                            args.batch_size, args.hidden_size, 
                            args.n_layers, args.dropout, args.residual,
                            args.lr, args.enforce_ratio, args.clip,
                            args.result_save_path)
    elif option=='c':
        model = torch.load(args.model_save_path)
    else:
        raise Exception("Eneter either r/c.")
        exit(1)
    
    if args.clear_prev_result and os.path.exists(args.result_save_path):
        os.remove(args.result_save_path)
    
    model.train(args.n_epochs, args.epoch_size, args.print_every)
    model.evaluate(args.n_eval_batches)
    torch.save(model, args.model_save_path)
