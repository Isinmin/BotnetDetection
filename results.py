import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, OrderedDict

algoritmos = ['Decision tree','Random forest','AdaBoost','NB Multinomial','NB Multivariate ','SVC Linear']
m1acurácias = [0.996604,0.99791,0.993208,0.894723,0.840125,0.958986]
m1n_caracteristicas = [6,5,9,4,10,5]

m1all_cols = ['duration','total_packets','iopr','total_bytes','avg_payload_length','bps','bpp','pps',
            'pct_packets_pushed','var_iat','avg_iat']

m1caracs = ['total_packets',
 'iopr',
 'total_bytes',
 'pps',
 'pct_packets_pushed',
 'avg_iat','total_packets', 'total_bytes', 'bps', 'bpp', 'pct_packets_pushed','duration',
 'iopr',
 'total_bytes',
 'avg_payload_length',
 'bpp',
 'pps',
 'pct_packets_pushed',
 'var_iat',
 'avg_iat','iopr', 'avg_payload_length', 'pct_packets_pushed', 'avg_iat','total_packets',
 'iopr',
 'total_bytes',
 'avg_payload_length',
 'bps',
 'bpp',
 'pps',
 'pct_packets_pushed',
 'var_iat',
 'avg_iat','duration', 'avg_payload_length', 'bpp', 'pct_packets_pushed', 'var_iat']

m1ocorrencias = Counter(m1caracs)
print(m1ocorrencias)

a = OrderedDict(m1ocorrencias.most_common())

m1cars = list(a.keys())
m1nums = list(a.values())

m1cars

algoritmos = ['Decision tree','Random forest','AdaBoost','NB Multinomial ','NB Multivariate ','SVC Linear']
m2acurácias = [0.9947753396029259,0.995559038662487,0.9963427377220481,0.9503657262277951,0.8699059561128527,0.9563740856844305]
m2n_caracteristicas = [7,39,16,17,10,12]

m2all_cols = ['total_fpackets', 'total_fvolume', 'total_bpackets', 'total_bvolume', 'min_fpktl', 'mean_fpktl', 'max_fpktl', 'std_fpktl', 'min_bpktl', 'mean_bpktl', 'max_bpktl', 'std_bpktl', 'min_fiat', 'mean_fiat', 'max_fiat', 'std_fiat', 'min_biat', 'mean_biat', 'max_biat', 'std_biat', 'duration', 'min_active', 'mean_active', 'max_active', 'fpsh_cnt', 'bpsh_cnt', 'total_fhlen', 'total_bhlen', 'total_bytes', 'total_packets', 'total_bits', 'bpp', 'bps', 'pps', 'var_iat', 'avg_iat', 'pct_packets_pushed', 'avg_payload_length', 'iopr']

m2caracs = ['total_fpackets', 'mean_fiat', 'max_biat', 'mean_active', 'fpsh_cnt', 'bpsh_cnt', 'pct_packets_pushed',
        'total_fpackets', 'total_fvolume', 't
        
        
        otal_bpackets', 'total_bvolume', 'min_fpktl', 'mean_fpktl', 'max_fpktl', 'std_fpktl', 'min_bpktl', 'mean_bpktl', 'max_bpktl', 'std_bpktl', 'min_fiat', 'mean_fiat', 'max_fiat', 'std_fiat', 'min_biat', 'mean_biat', 'max_biat', 'std_biat', 'duration', 'min_active', 'mean_active', 'max_active', 'fpsh_cnt', 'bpsh_cnt', 'total_fhlen', 'total_bhlen', 'total_bytes', 'total_packets', 'total_bits', 'bpp', 'bps', 'pps', 'var_iat', 'avg_iat', 'pct_packets_pushed', 'avg_payload_length', 'iopr',
       'total_fvolume', 'mean_fpktl', 'max_fpktl', 'mean_bpktl', 'max_bpktl', 'min_fiat', 'max_fiat', 'max_biat', 'max_active', 'bpsh_cnt', 'total_fhlen', 'bps', 'pps', 'pct_packets_pushed', 'avg_payload_length', 'iopr',
       'total_fpackets', 'total_bpackets', 'min_fpktl', 'min_bpktl', 'min_fiat', 'mean_fiat', 'max_fiat', 'std_fiat', 'min_biat', 'mean_biat', 'max_biat', 'std_biat', 'fpsh_cnt', 'bpsh_cnt', 'avg_iat', 'pct_packets_pushed', 'iopr',
       'min_fiat', 'mean_fiat', 'max_fiat', 'std_fiat', 'min_biat', 'mean_biat', 'max_biat', 'std_biat', 'var_iat', 'avg_iat',
        'min_fpktl', 'mean_fiat', 'max_fiat', 'std_fiat', 'mean_biat', 'max_biat', 'std_biat', 'bpp', 'avg_iat', 'pct_packets_pushed', 'avg_payload_length', 'iopr']

m2ocorrencias = Counter(m2caracs)

a = OrderedDict(m2ocorrencias.most_common())
m2cars = list(a.keys())
m2nums = list(a.values())

sns.set_context('paper', font_scale=2.9)
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
print(m2ocorrencias)

plt.figure(figsize=(21,15),frameon=False)
plt.title('Точность при использовании грубой силы')
plt.ylabel('Точность')
plt.xlabel('Алгоритмы')
plot = sns.barplot(x=algoritmos,y=m1acurácias,color='blue')

fig = plot.get_figure()

fig.savefig('точность1.png')

plt.figure(figsize=(21,15),frameon=False)
plt.title('Точность при использования метода регрессии')
plt.ylabel('Точность')
plt.xlabel('Алгоритмы')
plot = sns.barplot(x=algoritmos,y=m2acurácias,color='blue')

fig = plot.get_figure()

fig.savefig('точность2.png')

plt.subplots(figsize=(18,10))
ind = np.arange(len(algoritmos))
p1 = plt.bar(ind, m1n_caracteristicas,0.35)
p2 = plt.bar(ind + 0.35, m2n_caracteristicas,0.35)

plt.ylabel('Характеристики')
plt.title('Сравнение количества требуемых характеристик')
plt.xticks(ind + 0.15, algoritmos)
plt.yticks(np.arange(0,0.1,1))
plt.legend((p1[0], p2[0]), ('Брутфорс', 'Регрессия'))


plt.savefig('сравнение1.png', bbox_inches='tight')
plt.subplots(figsize=(18,10))
ind = np.arange(len(algoritmos))
p1 = plt.bar(ind, m1acurácias,0.35)
p2 = plt.bar(ind + 0.35, m2acurácias,0.35)

plt.ylabel('Точность')
plt.title('Точность классификаторов')
plt.xticks(ind + 0.15, algoritmos)
# plt.yticks(np.arange(0,0.1,1))
plt.legend((p1[0], p2[0]), ('Точность брутфорса', 'Точность регрессии'), bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)


plt.savefig('сравнениеточности2.png', bbox_inches='tight')
plt.figure(figsize=(20,10),frameon=False)
plt.title('Частота использования характеристики (брутфорс)')
plt.ylabel('Частота использования')
plt.xlabel('Характеристика')
plt.xticks(rotation=30, ha='right')
plot = sns.barplot(x=m1cars,y=m1nums,color='blue')

fig = plot.get_figure()

fig.savefig('характеристики1метод.png', bbox_inches='tight')

plt.figure(figsize=(20,10),frameon=False)
plt.title('Частота использования (регрессия)')
plt.ylabel('Частота использования ')
plt.xlabel('Характеристика')
plt.xticks(rotation=30, ha='right')
plot = sns.barplot(x=m2cars[:10],y=m2nums[:10],color='blue')

fig = plot.get_figure()

fig.savefig('характеристики2метод.png', bbox_inches='tight')
plt.figure(figsize=(20,10),frameon=False)
plt.title('Частота использования(регрессия)')
plt.ylabel('Частота использования')
plt.xlabel('Характеристика')
plt.yticks(np.arange(0,3,0.5))
plt.xticks(rotation=30, ha='right')
plot = sns.barplot(x=m2cars[::-1][:10][::-1],y=m2nums[::-1][:10][::-1],color='blue')

fig = plot.get_figure()

fig.savefig('Características Método2-menos.png', bbox_inches='tight')
a

test = [(key,value) for key,value in a.items() if key in m1cars]
test
svmcars = list(test.keys())
svmnums = list(test.values())
svmcars
plt.figure(figsize=(20,10),frameon=False)
plt.title('Частота характеристик из брутфорса в регрессии')
plt.ylabel('Количество вхождений')
plt.xlabel('Характеристики')
plot = sns.barplot(x=svmcars,y=svmnums,color='blue')
plt.xticks(rotation=30, ha='right')

fig = plot.get_figure()

fig.savefig('характеристикиСВМ.png', bbox_inches='tight')
sns.set_context('paper', font_scale=1.4)
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
accs = [0.996604,0.978840,0.749477]
nomes = ['Линейное','РБФ','Сигмоида']
plt.title('Точность метода опорных векторов')
plt.ylabel('Точноть')
plt.xlabel('Ядра')

plot = sns.lineplot(x=nomes,y=accs)

fig = plot.get_figure()

fig.savefig('гиперпараметы.png')
