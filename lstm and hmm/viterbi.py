# -*- coding:utf-8 -*-
def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    :param obs: 可见序列
    :param states: 隐状态
    :param start_p: 开始概率
    :param trans_p: 转换概率
    :param emit_p: 发射概率
    :return: 序列+概率
    """
    path = {}
    V = [{}]  # 记录第几次的概率
    for state in states:
        V[0][state] = start_p[state] * emit_p[state].get(obs[0], 0) #初始状态对应初始观测值的概率
        path[state] = [state]
    for n in range(1, len(obs)): #除了初始的观测值，从第1位置到len-1位置的观测值
        V.append({})  #新增加一个V数组空间
        newpath = {}
        for k in states: #求解从上个观测值转移到新观测值在新的各个不同状态下的最大概率
            pp,pat=max([(V[n - 1][j] * trans_p[j].get(k,0) * emit_p[k].get(obs[n], 0) ,j )for j in states])
            #最大的概率pp的之前的状态pat等于之前初始状态转移到新状态并且输出该观测词时的最大值
            #j和pat代表过去的状态，k代表新的状态
            V[n][k] = pp   #记录某个k状态对应的最大概率
            newpath[k] = path[pat] + [k] #记录最大概率下过去状态到新状态的路径
            #添加时就根据上一次的状态（不是最初状态）来决定新状态进入哪个状态序列当中
            #由于初始状态path[pat]里包含了值，因此就变成{sun:[sun,rain,sun],rain:[rain,sun,sun]}的形式
            # path[k] = path[pat] + [k]#不能提起变，，后面迭代好会用到！
        path=newpath    #这一轮观测值结束，path中存储的是之前所有的newpath[k],其形如{k:[sun,rain,sun],k':[rain,sun,sun]}
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    #最后再进行挑选最大的哪一个状态序列
    return prob, path[state]
#输出概率以及最大概率的状态路径