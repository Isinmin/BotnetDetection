import pandas as pd
import numpy as np

columns = ['srcip','srcport','dstip','dstport','proto','total_fpackets','total_fvolume','total_bpackets','total_bvolume',
'min_fpktl','mean_fpktl','max_fpktl','std_fpktl','min_bpktl','mean_bpktl','max_bpktl','std_bpktl','min_fiat','mean_fiat',
'max_fiat','std_fiat','min_biat','mean_biat','max_biat','std_biat','duration','min_active','mean_active','max_active',
'std_active','min_idle','mean_idle','max_idle','std_idle','sflow_fpackets','sflow_fbytes','sflow_bpackets','sflow_bbytes',
'fpsh_cnt','bpsh_cnt','furg_cnt','burg_cnt','total_fhlen','total_bhlen','dscp']

df = pd.read_csv('Datasets/testFlow.csv',names=columns)
df.head()
df = df[df['proto']==6]


df.columns[(df == 0).all()] # проверка нулевых значений
malicious_ips = [...] # ip зараженных узлов

irc_attacks = [...] # набор адресов (src,dst) irc-соединения
def flow_label(df):
    labels = []
    
    for index,data in df.iterrows():

        src = data['srcip']
        dst = data['dstip']
        

        if((src in malicious_ips) or (dst in malicious_ips)):
            labels.append(1)
        elif(((src,dst) in irc_attacks) or ((dst,src) in irc_attacks)):
            labels.append(1)
        else:
            labels.append(0)
    
    return labels

df['label'] = flow_label(df)

def flow_label(df):
    labels = []
    
    for index,data in df.iterrows():

        src = data['srcip']
        dst = data['dstip']
        

        if((src in malicious_ips) or (dst in malicious_ips)):
            labels.append(1)
        elif(((src,dst) in irc_attacks) or ((dst,src) in irc_attacks)):
            labels.append(1)
        else:
            labels.append(0)
    
    return labels

df['label'] = flow_label(df)
other_botnets = ['147.32.84.160','192.168.3.35', '192.168.3.25', '192.168.3.65', '172.29.0.116']
df = df[~df['srcip'].isin(other_botnets)]

df = df[~df['dstip'].isin(other_botnets)]

from collections import Counter

Counter(df['label'])
_underscore = 6379
_total = len(df[df['label'] == 0]) - _underscore #получение кол-ва элементов для удаления
_df_underscore_index = df[df['label'] == 0].head(_total).index #набор данных для удаления
df.drop(_df_underscore_index, inplace=True)# удаление поднабора данных из основного набора данных
df.reset_index(drop=True,inplace=True)
Counter(df['label'])

def flow_total_bytes():
    total_bytes = []
    for index,data in df.iterrows():

        bytes_forward = data['total_fvolume']
        bytes_backward = data['total_bvolume']

        bytes_sum = bytes_forward + bytes_backward

        total_bytes.append(bytes_sum)

    df['total_bytes'] = total_bytes
    
def flow_total_packets():
    total_packets = []
    for index,data in df.iterrows():

        packets_forward = data['total_fpackets']
        packets_backward = data['total_bpackets']

        packets_sum = packets_forward + packets_backward

        total_packets.append(packets_sum)

    df['total_packets'] = total_packets
    
def flow_total_bits():
    total_bits = []
    for index,data in df.iterrows():

        total_bytes = data['total_bytes']

        bits = total_bytes * 8

        total_bits.append(bits)

    df['total_bits'] = total_bits

def bytes_per_packet():
    bpp = []
    for index,data in df.iterrows():

        _bytes = data['total_bytes']
        packets = data['total_packets']
        val = _bytes / packets

        bpp.append(val)

    df['bpp'] = bpp
    
def bits_per_sec():
    bps = []
    for index,data in df.iterrows():

        bits = data['total_bytes'] * 8
        secs = data['duration'] * 0.000006
        
        if secs != 0:
            val = bits/secs
        else:
            val = 0

        bps.append(val)

    df['bps'] = bps   

def packets_per_sec():
    pps = []
    for index,data in df.iterrows():

        packets = data['total_packets']
        secs = data['duration'] * 0.000006

        if secs != 0:
            val = packets/secs
        else:
            val = 0

        pps.append(val)

    df['pps'] = pps   
    
def avg_var_iat():
    iat = []
    for index,data in df.iterrows():

        f_iat = data['std_fiat'] 
        b_iat = data['std_biat']

        f_iat = f_iat * f_iat
        b_iat = b_iat * b_iat

        avg = (f_iat + b_iat)/2

        iat.append(avg)

    df['var_iat'] = iat
    
def avg_iat():
    iat = []
    for index,data in df.iterrows():

        f_iat = data['mean_fiat'] 
        b_iat = data['mean_biat']

        avg = (f_iat + b_iat)/2

        iat.append(avg)

    df['avg_iat'] = iat
    
def pct_packets_pushed():
    pctpp = []
    for index,data in df.iterrows():
        
        packets_pushed = data['total_fpackets']
        total_packets = data['total_packets']
        
        if total_packets != 0:
            val = packets_pushed/total_packets
        else:
            val = 0
        
        pctpp.append(val)
    
    df['pct_packets_pushed'] = pctpp
    
def iopr():
    iopr = []
    for index,data in df.iterrows():
        
        packets_pushed = data['total_fpackets']
        packets_pulled = data['total_bpackets']
        
        if packets_pushed != 0:
            val = packets_pulled/packets_pushed
        else:
            val = 0
            
        iopr.append(val)
        
    df['iopr'] = iopr
    
def avg_payload_length():
    # (bytes_header_forward + bytes_header_back) - total_bytes = payload length
    # payload_length / packets = average payload length
    avg_pl = []
    for index,data in df.iterrows():
        
        header_f = data['total_fhlen']
        header_b = data['total_bhlen']
        total_b = data['total_bytes']
        packets = data['total_packets']
        
        if packets != 0:
            payload_length = total_b - (header_b + header_f)
            avg = payload_length / packets
        else:
            avg = 0
        
        avg_pl.append(avg)
    
    df['avg_payload_length'] = avg_pl
    
flow_total_bytes()
flow_total_packets()
flow_total_bits()
bytes_per_packet()
bits_per_sec()
packets_per_sec()
avg_var_iat()
avg_iat()
packets_per_sec()
pct_packets_pushed()
avg_payload_length()
iopr()

df.columns

df.to_csv('Datasets/trainset.csv')
from sklearn.lineral_model import LinearRegression
from sklearn.metrics import accuracy_score

linearReg = LinearRegression()
linear
