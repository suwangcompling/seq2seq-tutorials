# seq2seq-tutorials
A. Building seq2seq models with Tensorflow (v1.0.0)

* seq2seq-tutorial01: basic architecture setup.
* seq2seq-tutorial02: Sutskever et al. (2014).
* seq2seq-tutorial03: Bahdanau et al. (2015).
* Enc-Dec-for-sorting-strings: Bahdanau et al. (2015).
* Ptr-Net-for-sorting-strings: Vinyals et al. (2016).
* Ptr-Net-for-sorting-sentences: Gong et al. (2016)

B. Building seq2seq models with PyTorch (v0.3.1)

* PyTorch-seq2seq-full-tutorial: Prakash et al. (2016), with toy data.
  * Bidirectional
  * Stacking
  * Residual links
  * Attention (Bahdanau or Luong)
* PyTorch-seq2seq-mscoco* (data = MSCOCO): Prakash et al. (2016)
  * Base: greedy search.
  * -beam: beam search.
* PyTorch-seq2seq-transformer: Vaswani et al. (2017)

NB: the PyTorch code uses a few helpers, which are stored in utils.py and seq2seq_data_loader.py
NB: to download the data used in the demos, go to https://github.com/iamaaditya/neural-paraphrase-generation/tree/dev/data, create a dir "mscoco/", and put the following files under it: train_source.txt, train_target.txt, test_source.txt, test_target.txt.
