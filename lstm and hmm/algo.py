from viterbi import viterbi
import pandas as pd
from pandas import DataFrame
state_list = ['B-BANK', 'I-BANK', 'B-PRODUCT', 'I-PRODUCT', 'B-COMMENTS_N', 'I-COMMENTS_N', 'B-COMMENTS_ADJ', 'I-COMMENTS_ADJ',
			   'O']
'''
B-BANK 代表银⾏实体的开始
I-BANK 代表银⾏实体的内部
B-PRODUCT 代表产品实体的开始
I-PRODUCT 代表产品实体的内部
O 代表不属于标注的范围
B-COMMENTS_N 代表⽤户评论（名词）
I-COMMENTS_N 代表⽤户评论（名词）实体的内部
B-COMMENTS_ADJ 代表⽤户评论（形容词）
I-COMMENTS_ADJ 代表⽤户评论（形容词）实体的内部
'''
# def viterbi(obs,states,start_p,trans_p,emit_p):
#     """
#     :param obs: 可见序列
#     :param states: 隐状态
#     :param start_p: 开始概率
#     :param trans_p: 转换概率
#     :param emit_p: 发射概率
#     :return: 序列+概率
#     """

#提取三个字典里的信息
file0=open('start.txt','r')
start_c=eval(file0.read())
file1=open('emit.txt','r',encoding='utf8')
emit_c=eval(file1.read())
file2=open('tran.txt','r',encoding='utf8')
trans_c=eval(file2.read())  #命名实体识别的文件

file0=open('output_classification.txt','r')
lable=eval(file0.read())  #情感分类的文件

filepath="test.csv" #打开训练文件，要放在当前目录下
algo_test=pd.read_csv(filepath)

file3=open('output.txt','w',encoding='utf8')

out = 'id' + ',' + 'BIO_anno' + ',' + 'class'+'\n'
file3.write(str(out))
for line in range(0,len(algo_test["text"])): 
    words_test = [one for one in algo_test["text"][line]]
    prop,out_state=viterbi(words_test,state_list,start_c,trans_c,emit_c)

    out_state = str(out_state)
    out_state = out_state.replace('\'', '')#转义字符，去除单引号
    out_state = out_state.replace(',', '')#转义字符，去除逗号
    out_state=out_state.strip('[')  #去除左右的[]
    out_state=out_state.strip(']')

    out = str(line) + ',' + out_state + ',' + lable[line]+'\n'
##将预测的编码序列写入新的文件
    file3.write(str(out))
file3.close()




