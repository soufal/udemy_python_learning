from __future__ import division
import pandas as pd
import os
import numpy as np
import math
import time
start =time.clock()


trainFile = r'adult.csv'
pwd = os.getcwd()
#os.chdir(os.path.dirname(trainFile))
trainData = pd.read_csv(os.path.basename(trainFile),header=None,dtype='int')
os.chdir(pwd)


groups1 = trainData.groupby([0,1,2,3,4,5,6,7,8,9,10]).groups
groups2 = trainData.groupby([1,2,3,4,5,6,7,8,9,10]).groups
groups3 = trainData.groupby([0,2,3,4,5,6,7,8,9,10]).groups
groups4 = trainData.groupby([0,1,3,4,5,6,7,8,9,10]).groups
groups5 = trainData.groupby([0,1,2,4,5,6,7,8,9,10]).groups
groups6 = trainData.groupby([0,1,2,3,5,6,7,8,9,10]).groups
groups7 = trainData.groupby([0,1,2,3,4,6,7,8,9,10]).groups
groups8 = trainData.groupby([0,1,2,3,4,5,7,8,9,10]).groups
groups9 = trainData.groupby([0,1,2,3,4,5,6,8,9,10]).groups
groups10 = trainData.groupby([0,1,2,3,4,5,6,7,9,10]).groups
groups11 = trainData.groupby([0,1,2,3,4,5,6,7,8,10]).groups
groups12 = trainData.groupby([0,1,2,3,4,5,6,7,8,9]).groups

groupsd = trainData.groupby([11]).groups


def e(index,groups):
    for item in groups.values():
        if index in item:
            return set(item)

		
def kd(index, group1, group2):
    s1=e(index,group1)
    s2=e(index, group2)
    return len(s1.union(s2))-len(s1.intersection(s2))
	
def_data=len(trainData)
def ai(len_data,group1,group2):
    _sum=0
    for i in range(len_data):
        _sum+=kd(i,group1,group2)
        return _sum/(len_data*len_data)

		

		
group=[groups2,groups3,groups4,groups5,groups6,groups7,groups8,groups9,groups10,groups11,groups12]

def nor(len_data,group):
    sum_= 0
    w=[]
    for j in group:
        sum_+=ai(len_data,groups1,j)
    for j in group:
        w.append(ai(len_data,groups1,j)/sum_)
    return w
w = nor(len(trainData),group)



def coe(c1,c2,c3,c4,c5,c6):
    if 0.5*(c2-c3)*(c5-c6)>=(c1-c2)*(c4-0.5*c5):
        return c1,c2,c3,c4,c5,c6



def uti(a_,b,c0):
    utia=-a_
    utip=math.log(2,math.e)/(math.log((b-a_),math.e)-math.log((c0-a_),math.e))
    utiq=1/(b-a_)**utip
    return utia,utip,utiq
	
def uti(a_,b,c0):
    utia=-a_
    utip=math.log(2,math.e)/(math.log(abs(b-a_),math.e)-math.log(abs(c0-a_),math.e))
    utiq=1/(b-a_)**utip
    return utia,utip,utiq

uti_list = []
a__b_c0_list = [(17,90,65),(1,8,5),(1,16,10),(1,16,9),(1,6,4),(1,14,9),(1,6,5),(1,5,3.2),(1,2,1.8),(1,99,55),(1,41,30)]
for  a_,b,c0 in a__b_c0_list:
    uti_list.append(uti(a_,b,c0))
uti_list



c1,c2,c3,c4,c5,c6 = 9,8,1,9,7,3
result = []
def wuti(c1,c2,c3,c4,c5,c6,i,w,columns):
    '''uuu'''
    coe(c1,c2,c3,c4,c5,c6)
    upp,ubp,unp,unn,ubn,upn,sumu=0,0,0,0,0,0,0
    for col in range(len(columns)):
#         print(' i  am here')
#         print(i,col)
        tmp = trainData.ix[i,columns[col]]-uti_list[col][0]
        sumu+=uti_list[col][2]*(tmp)**uti_list[col][1]
#         print('over')

    for col in range(len(columns)):
        tmp = (uti_list[col][2]*(trainData.ix[i,columns[col]]-uti_list[col][0])**uti_list[col][1])
        tmp1 = np.power(c1,tmp) + c1
        tmp2 = np.power(c2,tmp) + c2
        tmp3 = np.power(c3,tmp) + c3
        tmp4 = np.power(c4,tmp) + c4
        tmp5 = np.power(c5,tmp) + c5
        tmp6 = np.power(c6,tmp) + c6
        upp+=w[col]*tmp1 / 18
        ubp+=w[col]*tmp2 / 18
        unp+=w[col]*tmp3/ 18
        unn+=w[col]*tmp4 / 18
        ubn+=(1/(sumu/len(columns)+1))*w[col]*tmp5 / 18
        upn+=(1/(sumu/len(columns)+1))*w[col]*tmp6 / 18

    alpha=(ubn-upn)/(upp-ubp+ubn-upn)
    beta=(unn-ubn)/(ubp-unp+unn-ubn)

    return upp,ubp,unp,unn,ubn,upn,alpha,beta
	
	

rough_mem_func = []
for key in groups1.keys():
    rough_mem_func.append(len(set(groupsd[1]).intersection(groups1[key])) / len(groups1[key]))


classes = [] 
for key in groups1.keys():
    classes.append(groups1[key][0])
    

columns=[0,1,2]
pos,bnd,neg = [],[],[]
for i in classes:
    result.append(wuti(c1,c2,c3,c4,c5,c6,i,w,columns))
    alpha,beta = result[-1][-2:]
    keys = list(groups1.keys())
    for i in range(len(rough_mem_func)):
        tmp = rough_mem_func[i]
        if tmp >= alpha:
            pos.extend(groups1[keys[i]])
        elif tmp <= beta:
            neg.extend(groups1[keys[i]])
        else:
            bnd.extend(groups1[keys[i]])

pos = set(pos)
bnd = set(bnd)
neg = set(neg)

correction_rate=(len(set(groupsd[1]).intersection(set(pos)))+len((set(groupsd[2])).intersection(set(neg))))/len(trainData)
print(correction_rate)

end = time.clock()
print('Running time: %s Seconds'%(end-start))

