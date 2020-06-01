# -*- coding: utf-8 -*-
# @Author  : zhangcy
# @Email   : 604641446@qq.com
# @Time    : 2020/5/29 16:35
import pyhttp
import json
import random,copy
pyserviceUrl='http://localhost:5000/'
serviceurl='jt_pyservice' #jt_kmeans,jt_Agglomerative,,jt_dbscan,jt_GMM,jt_brich,jt_mean_shfit,jt_Affinity_propa,jt_StandardSca
url=pyserviceUrl+serviceurl

data = []
# lofar data generator
def lofar_normal_sample_generate():
    normal_sample = []
    temp_lenth = random.randint(15, 25)
    for i in range(temp_lenth):
        point = [random.randint(1, 3000), random.randint(1, 100) / 100]
        normal_sample.append(point)
    return normal_sample


def data_lofar_generate(class_count):
    data = []
    data_normal = []
    data_empty = []
    for m in range(class_count):
        data_empty.append(data_normal.copy())

    for i in range(class_count):
        data_normal.append(lofar_normal_sample_generate())
        for j in range(20):
            temp = copy.deepcopy(data_normal[i])
            for k in range(len(temp)):
                temp[k][0] = temp[k][0] + random.randint(1, 20)
                temp[k][1] = temp[k][1] * random.uniform(0.75, 1)
            for o in range(random.randint(1, 5)):
                pop_index = random.randint(0, len(temp) - 1)
                temp.pop(pop_index)
            data_empty[i].append(temp)

    for l in range(class_count):
        data.extend(data_empty[l])
    return data
ll = data_lofar_generate(5)
print(ll)

 #choose clusteing method:k-means;DBSCAN;GMM;BRICH;FCM;SC
parameters = {'data': ll, 'thrhd': 10, 'K': 5, 'method': 'SC'}
res=pyhttp.getmodel(url,parameters)

print(res)