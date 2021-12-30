# lstm-hmm
nlp大作业
任务1：命名实体识别
任务2：情感分类标签预测
给定train.csv文件和test.csv文件，训练文件中有银行评论，以及对应的每个字的命名和该评论对应的情感标签(0,1,2),测试文件中只有银行评论

任务1：进行命名实体识别，利用HMM模型，通过viterbi算法实现/利用LSTM模型进行命名实体识别
任务2：利用word2vector库进行词的embedding，然后通过LSTM模型进行预测

任务1相关文件：

dictionary.py:执行得到HMM模型中的start,emit和tran字典

viterbi:viterbi一般算法

emit.txt:生成的emit字典

start.txt:生成的start字典

tran.txt:生成的tran字典

任务2相关文件：
LSTM_emotion_classify.py:第二个任务的所有代码，最终生成splite_word_all.txt和classification_model.txt和output_classification.txt文件，其先利用jieba对训练集和测试集的每个句子分词，然后利用word2vector库做embedding，并生成模型，然后利用LSTM训练和预测。

emotion_classify copy.ipynb:和上面的py文件代码一样，不过是运行在jupyter上的

test_with_class.csv:为了与代码中文件处理一致性和算法中的数据兼容性，将test增加类别标签，并全设置为0

splite_word_all.txt:生成的分词文件

classification_model.txt:word2vector库做embedding生成的模型

output_classification.txt:最后得到的test中各个评论标签预测结果


algo.py用于完成两个任务，output.txt存放最终结果

testoutput.ipynb:用于任务1检测输出和测试集中每评论字符数对应的命名实体数是否相等

执行顺序：

1.执行dictionary.py生成字典

2.执行LSTM_emotion_classify.py生成标签预测结果

3.执行algo.py得到最终output.txt结果

4.在获得output.txt后，在jupyter上执行testoutput.ipynb，测试输出是否有问题，并且将输出格式转化为int，string，int形式，输出到output.csv文件夹中


我还增加了一个lstm_ner文件夹，单独做LSTM的命名实体识别任务，代码在lstm_ner.ipynb中，按照次序执行即可得到新的output1.csv文件，该文件是在之前用HMM做NER和LSTM做情感标签预测任务得到的output1.csv的基础上，修改BIO_anno列而生成的新文件，只改变了BIO_anno。

最后的尝试，因为train.csv文件中情感标签为2的评论数量很多，因此我利用lstm模型训练的过程中，除了少数的几次会发现训练集精度随着epoch的增加而下降，其他情况都是训练集精度不变，因此我怀疑是发生的过拟合，于是我删除了许多标签为2的评论，并且将新的训练集命名为train_xiaochu2.csv，再用新的emotion_classify_copy2.ipynb进行训练以及output_classification.txt的产生，再运行algo.py和testoutput.ipynb得到最终的结果output.csv，此时最终结果的评分上升了0.02.

