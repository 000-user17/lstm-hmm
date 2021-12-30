# -*- coding:utf-8 -*-
import sys
import pandas as pd
from pandas import DataFrame

"""
计算HMM开始概率字典，开始概率字典，开始概率字典’
并输出为txt文件
"""
start_c={}#开始概率字典 ，即开始状态的概率
transport_c={}#开始概率字典，即从一个状态转移到另一个状态的概率
emit_c={}#开始概率字典，即在一个状态下发射一个观测值概率

state_list = ['B-BANK', 'I-BANK', 'B-PRODUCT', 'I-PRODUCT', 'B-COMMENTS_N', 'I-COMMENTS_N', 'B-COMMENTS_ADJ', 'I-COMMENTS_ADJ',
			   'O']
for state0 in state_list:
    transport_c[state0]={}  #初始化transport_c[state0]数组，其内包含的是从transport[state0]转向其他所有（包括自身）的概率集合
    for state1 in state_list:
        transport_c[state0][state1]=0.0  #state0 -> state 1 的概率，初始化概率为0
    emit_c[state0]={}   
    start_c[state0]=0.0   #开始概率
vocabs=[] #所有的字，标点符号和数字表
classify=[]  #所有的bio类别
class_count={}
for state in state_list:
    class_count[state]=0.0  #初始state计数都为0
    
filepath="train.csv" #打开训练文件，要放在当前目录下
df=pd.read_csv(filepath)

##下面的循环在于填充emit_c和transport_c
for line in range(0,len(df["text"])):  #对于训练集中所有的行
    bios = df["BIO_anno"][line].split( ) #将每个分词根据空格进行分割
    words = [one for one in df["text"][line]]#把每一行分解为多个字以及标点符号，因为每个句子的BIO等于标点符号加数字加字的个数
        
    for word in words:
        vocabs.append(word)
    for bio in bios:
        classify.append(bio)

    else:
        for n in range(0,len(vocabs)):
            class_count[classify[n]]+=1.0   #如果有一个相应的bio,则其值加1，相当于哈希表
            if vocabs[n] in emit_c[classify[n]]:
                emit_c[classify[n]][vocabs[n]] += 1.0  
                #如果某个word在发射概率中，即在某个bio下转移到某个word的概率已经存在了，则加1
            else:
                emit_c[classify[n]][vocabs[n]] = 1.0
                #如果不存在，便从无设置为1
            if n==0:
                start_c[classify[n]] += 1.0  #根据每一行的初始状态设置初始概率
                #如果
            else:
                transport_c[classify[n-1]][classify[n]]+=1.0
                #根据每一行的转移状态设置
    vocabs = []
    classify = []  #清零，进行下一行的计数
    
for state in state_list:#将各个概率归一化，使其各个值相加起来和为1
    start_c[state]=start_c[state]*1.0/df["text"].size  
    #用start_c除以整体的行数来归一化
    for li in emit_c[state]:
        emit_c[state][li]=emit_c[state][li]/class_count[state]
    for li in transport_c[state]:
        transport_c[state][li]=transport_c[state][li]/class_count[state]
    
file0=open('start.txt','w',encoding='utf8')
file0.write(str(start_c))
file1=open('tran.txt','w',encoding='utf8')
file1.write(str(transport_c))
file2=open('emit.txt','w',encoding='utf8')
file2.write(str(emit_c))
file0.close()
file1.close()
file2.close()


