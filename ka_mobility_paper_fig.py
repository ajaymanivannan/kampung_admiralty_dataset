#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:54:30 2022

@author: aj
"""
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from itertools import groupby
import string
import random
from operator import itemgetter
import datetime
import networkx as nx
from collections import defaultdict

import seaborn as sns
import json
#%%

G=nx.read_gexf('../data/ka_network_graph_a_17mar2021_with_mobility.gexf') 
#G=nx.read_gexf('ka_network_graph_a_17mar2021_with_mobility.gexf') 

print('Number of nodes: ',len(G.nodes())) 
print('Number of edges: ',len(G.edges())) 

#%% Load mobility data per edge


df_mobility = pd.read_csv('../data/trip_data_ka_dataset_aj_18mar2021.csv')
print(df_mobility.head())
print(df_mobility.columns.values)
#%% Group nodes by floor group



node_levels = defaultdict(list)
for n,d in G.nodes(data=True):
    node_levels[d["level"]].append(n)
    
groups = {
          "carparks": node_levels["B2"]+node_levels["B1"],
          "L1": node_levels["L1"],
          "L2": node_levels["L2"],
          "L3-L4": node_levels["L3"]+["L4_CLL"],
          "L6": node_levels["L6"],
         }

r7 = ['L7_R1_LL', 'L7_R1_LL_corridor', 'L7_R2_LL_corridor', 'L7_R2_LL']
c7 = [ n for n in node_levels["L7"] if n not in r7 ]
groups["L7"] = c7

r8 = ['L8_R1_LL', 'L8_R1_LL_corridor', 'L8_R2_LL_corridor', 'L8_R2_LL']
c8 = [ n for n in node_levels["L8"] if n not in r8 ]
r9 = ['L9_R1_LL', 'L9_R1_LL_corridor', 'L9_R2_LL_corridor', 'L9_R2_LL']
c9 = [ n for n in node_levels["L9"] if n not in r9 ]
groups["rooftop_garden"] = c8 + c9

r4 = [ n for n in node_levels["L4"] if not n=="L4_CLL"]
r5 = node_levels["L5"]
r6 = ['L6_Deck_1', 'L6_Deck_2', 'L6_Deck_3', 'L6_Deck_4', 'L6_R1_LL_corridor', 'L6_R1_LL', 'L6_Link_bridge', 'L6_R2_LL_corridor', 'L6_R2_LL']
r10 = node_levels["L10"]
r11 = node_levels["L11"]
groups["residential"] = r4+r5+r6+r7+r8+r9+r10+r11

group_levels = ['carparks', 'L1', 'L2', 'L6', 'rooftop_garden', 'residential']
groups2 = { k:groups[k] for k in group_levels }
temp = []
oth = [ k for k in groups.keys() if not(k in group_levels) ]
for k in oth:
    ns = groups[k]
    temp.extend(ns)
#print(temp)
groups2["other"] = temp
fg=[]
for key,value in groups2.items():
    for v in value:
        fg.append([key,v])

df_fg = pd.DataFrame.from_records(fg,columns=['fg','zone'])
#%%
for n,d in G.nodes(data=True):
    print(n,d)
    break
#%% Add floor group as node attribute to existing Graph


print('Adding node attributes....')
i=0
node_attr_dic={}
for node in G.nodes():
    node_filtered=df_fg[df_fg['zone']==node]       
    if len(node_filtered)==0:
        floor_group=''
    else:
        floor_group=list(node_filtered['fg'])[0]
    #add it to dic
    if not node in node_attr_dic:
        node_attr_dic[node]={}
    node_attr_dic[node]={'floor_group':floor_group}         
#use the created dictionary to assign node attributes to graph
nx.set_node_attributes(G, node_attr_dic) 

#%% create new graph - floor group

# G_fg=nx.DiGraph()
# fg_node=nx.get_node_attributes(G,'floor_group')
# for e in G.edges():
#     i1=e[0]
#     i2=e[1]
#     mn=G.get_edge_data(i1,i2)['mobility_number']
#     if mn>0:
#         fg_i1=fg_node[i1]
#         fg_i2=fg_node[i2]
#         if not G_fg.has_edge(fg_i1, fg_i2):
#             G_fg.add_edge(fg_i1, fg_i2,mobility_number=mn)
#         else:
#             G_fg[fg_i1][fg_i2]["mobility_number"]+=mn
    
# for u,v,d in G_fg.edges(data=True):
#     print(u,v,d)    
# nx.write_gexf(G_fg,'../../data/KA experiment/post_processed_data/ka_network_graph_a_floor_group_5april2022_directed_edges.gexf')
# print('Graph is stored')  
#%%  create new graph -  location type group

# G_lt=nx.DiGraph()
# fg_node=nx.get_node_attributes(G,'location_type')
# for e in G.edges():
#     i1=e[0]
#     i2=e[1]
#     mn=G.get_edge_data(i1,i2)['mobility_number']
#     if mn>0:
#         fg_i1=fg_node[i1]
#         fg_i2=fg_node[i2]
#         if not G_lt.has_edge(fg_i1, fg_i2):
#             G_lt.add_edge(fg_i1, fg_i2,mobility_number=mn)
#         else:
#             G_lt[fg_i1][fg_i2]["mobility_number"]+=mn
    
# for u,v,d in G_lt.edges(data=True):
#     print(u,v,d)    
# nx.write_gexf(G_lt,'../../data/KA experiment/post_processed_data/ka_network_graph_a_function_group_5april2022_directed_edges.gexf')
# print('Graph is stored') 

#%% Create graph for floor group
floor_lab2 = {
    "carparks":"Carparks", 
    "L1":"L1",
    "L2":"L2",
    "L6":"L6",
    "rooftop_garden":"Rooftop garden",
    "residential":"Residential",
}
G_od_fg=nx.DiGraph()
fg_node=nx.get_node_attributes(G,'floor_group')
self_loop={}
for index,row in df_mobility.iterrows():
    if row['yes_no']=='yes':
        i1=row['source']
        i2=row['target']

        fg_i1=fg_node[i1]
        fg_i2=fg_node[i2]
        fg_i1=floor_lab2[fg_i1]
        fg_i2=floor_lab2[fg_i2]
        # if fg_i1==fg_i2:
        #     if not fg_i1 in self_loop:
        #         self_loop[fg_i1]=0
        #     self_loop[fg_i1]+=1
        #     continue
        if not G_od_fg.has_edge(fg_i1, fg_i2):
            G_od_fg.add_edge(fg_i1, fg_i2,mobility_number=1)
        else:
            G_od_fg[fg_i1][fg_i2]["mobility_number"]+=1  
            
for u,v,d in G_od_fg.edges(data=True):
    print(u,v,d)    
  
# self_loop['Rooftop garden']=0

# print('Adding node attributes....')
# i=0
# node_attr_dic={}
# for node in G_od_fg.nodes():
#     if not node in node_attr_dic:
#         node_attr_dic[node]={}
#     node_attr_dic[node]={'internal_mobility_number':self_loop[node]}         
# #use the created dictionary to assign node attributes to graph
# nx.set_node_attributes(G_od_fg, node_attr_dic) 


nx.write_gexf(G_od_fg,'ka_network_graph_a_floor_group_od_5april2022_directed_edges.gexf')
print('Graph is stored')
#%% graph for location group
lt_lab2 = {
 'Community_Street':'Community',
 'Garden_Street':'Garden',
 'Residential_Street':'Residential',
 'Commercial_Street':'Commercial',
 'Social_Space':'Social',
 'Corridor':'Corridor',
 'Vertical_Street':'Vertical',
 'Entrance_Street':'Entrance'}
G_od_lt=nx.DiGraph()
lt_node=nx.get_node_attributes(G,'location_type')
self_loop={}
for index,row in df_mobility.iterrows():
    if row['yes_no']=='yes':
        i1=row['source']
        i2=row['target']

        lt_i1=lt_node[i1]
        lt_i2=lt_node[i2]
        lt_i1=lt_lab2[lt_i1]
        lt_i2=lt_lab2[lt_i2]
        # if lt_i1==lt_i2:
        #     if not lt_i1 in self_loop:
        #         self_loop[lt_i1]=0
        #     self_loop[lt_i1]+=1
        #     continue
        if not G_od_lt.has_edge(lt_i1, lt_i2):
            G_od_lt.add_edge(lt_i1, lt_i2,mobility_number=1)
        else:
            G_od_lt[lt_i1][lt_i2]["mobility_number"]+=1  
            
for u,v,d in G_od_lt.edges(data=True):
    print(u,v,d)    
G_od_lt.add_node('Residential')  
# self_loop['Residential']=0
# self_loop['Corridor']=0

# print('Adding node attributes....')
# i=0
# node_attr_dic={}
# for node in G_od_lt.nodes():
#     if not node in node_attr_dic:
#         node_attr_dic[node]={}
#     node_attr_dic[node]={'internal_mobility_number':self_loop[node]}         
# #use the created dictionary to assign node attributes to graph
# nx.set_node_attributes(G_od_lt, node_attr_dic) 


nx.write_gexf(G_od_lt,'ka_network_graph_a_floor_group_lt_5april2022_directed_edges.gexf')
print('Graph is stored')
#%%

pos_floor_lab = {
 'Carparks': (0.25, 1.08),
 'L1': (0.13, 0.7),
 'L2': (-0.25, 0.26),
 'L6': (-0.12, -0.65),
 'Rooftop garden': (-0.3, -1.1),
 'Residential': (0.28, -0.3)}
pos_cat_lab = {
 'Community': [-0.63, 0.24],
 'Garden': [-0.64, -0.35],
 'Residential': [0.28, -1.0],
 'Commercial': [0.17, 0.68],
 'Social': [0.34, -0.32],
 'Corridor': [-0.4, 0.6],
 'Vertical': [-0.08, -0.28],
 'Entrance': [0.63, 0.48]}

#%%  Network graph visual - by group
fg_node=nx.get_node_attributes(G_od_fg,'internal_mobility_number')
clrs = sns.color_palette("tab10")
group_levels=['Residential', 'L1', 'Rooftop garden', 'L2', 'Carparks', 'L6' ]
nsize_floor = []
for n in group_levels:
    if G_od_fg.has_edge(n,n):
        edw = G_od_fg[n][n]["mobility_number"]
    else:
        edw = 1
    edw = edw*10
    nsize_floor.append(edw)
    print(n, edw)
n_clrs_level = [ clrs[i] for i in range(len(group_levels)) ]


with open('pos_files/pos_floor_group.json', 'r') as fp:
    pos_floor = json.load(fp)

  
# pos_floor['Rooftop Garden']=[0.1, -0.5]
# pos_floor['Carparks']=[0,1]
floor_lab2 = {
    "Carparks":"Car-parks", 
    "L1":"Level-1",
    "L2":"Level-2",
    "L6":"Level-6",
    "Rooftop garden":"Rooftop garden",
    "Residential":"Residential",
}


fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")
ax1,ax2 = axs
nx.draw_networkx_nodes(G_od_fg, nodelist=group_levels, pos=pos_floor, ax=ax1, node_size=nsize_floor, node_color=n_clrs_level)
edgelist = G_od_fg.edges()
edgeweight = [ G_od_fg[u][v]["mobility_number"] for u,v in edgelist ]
edgewidth = [ np.log(w)*1+1 if w>0 else 1 for w in edgeweight ]
nx.draw_networkx_edges(G_od_fg, pos=pos_floor, ax=ax1, edgelist=edgelist, width=edgewidth, 
                       connectionstyle="arc3,rad=0.15", edge_color="xkcd:navy", alpha=.6, min_target_margin=15, min_source_margin=15)
nx.draw_networkx_labels(G_od_fg, pos=pos_floor_lab, labels=floor_lab2, ax=ax1)
ax1.set_title("(a) Between floor groups", loc="left")

#function group
#lt_node=nx.get_node_attributes(G_od_lt,'internal_mobility_number')
loc_listb = ['Residential_Street', 'Social_Space', 'Garden_Street', 'Commercial_Street', 'Entrance_Street', 'Community_Street', 'Vertical_Street', 'Corridor']
loc_listb2 = [ a.split("_")[0] for a in loc_listb ]
n_clrs_cat = [ clrs[i] for i in range(len(loc_listb2)) ] 
nsize_cat = []
for n in loc_listb2:
    if G_od_lt.has_edge(n,n):
        edw = G_od_lt[n][n]["mobility_number"]
    else:
        edw = 1
    edw = edw*10
    nsize_cat.append(edw)
    print(n, edw)

   

with open('pos_files/pos_cat.json', 'r') as fp:
    pos_cat = json.load(fp)  
    
nx.draw_networkx_nodes(G_od_lt, nodelist=loc_listb2, pos=pos_cat, ax=ax2, node_size=nsize_cat, node_color=n_clrs_cat)
edgelist = G_od_lt.edges()
edgeweight = [ G_od_lt[u][v]["mobility_number"] for u,v in edgelist ]
edgewidth = [ np.log(w)*1+1 if w>0 else 1 for w in edgeweight ]
nx.draw_networkx_edges(G_od_lt, pos=pos_cat, ax=ax2, edgelist=edgelist, width=edgewidth, 
                        connectionstyle="arc3,rad=0.15", edge_color="xkcd:navy", alpha=.6, min_target_margin=15, min_source_margin=15)
nx.draw_networkx_labels(G_od_lt, pos=pos_cat_lab, ax=ax2)
ax2.set_title("(b) Between location categories", loc="left")

plt.tight_layout()
plt.savefig("cross_groups_cats.png", dpi=300, bbox_inches="tight")          
        
    
#%% O-D matrix

floor_order = [ 'Carparks', 'L1', 'L2', 'L6', 'Rooftop garden', 'Residential' ]
floor_order2 = [ 'Car-parks', 'Level-1', 'Level-2', 'Level-6', 'Rooftop garden', 'Residential' ]
cat_order = ['Entrance', 'Social', 'Commercial', 'Community', 'Garden', 'Residential',  'Vertical', 'Corridor']

mat_floor = nx.to_numpy_matrix(G_od_fg, weight="mobility_number", nodelist=floor_order)
print(mat_floor)
diag_floor = np.copy(np.diag(mat_floor))
np.fill_diagonal(mat_floor, 0.)
print(mat_floor, diag_floor)

mat_cat = nx.to_numpy_matrix(G_od_lt, weight="mobility_number", nodelist=cat_order)
print(mat_cat)
diag_cat = np.copy(np.diag(mat_cat))
np.fill_diagonal(mat_cat, 0.)
print(mat_cat, diag_cat)

#mat_floor, mat_floor.sum(axis=1)
rowsum = mat_floor.sum(axis=1)
mat_floor2 = np.matrix(np.zeros(mat_floor.shape))
for i in range(mat_floor.shape[0]):
    for j in range(mat_floor.shape[1]):
        v = mat_floor[i,j]
        b = rowsum[i]
        #print(v,b)
        mat_floor2[i,j] = v/b
#mat_floor, mat_floor2

rowsum = mat_cat.sum(axis=1)
mat_cat2 = np.matrix(np.zeros(mat_cat.shape))
for i in range(mat_cat.shape[0]):
    for j in range(mat_cat.shape[1]):
        v = mat_cat[i,j]
        b = rowsum[i]
        #print(v,b)
        mat_cat2[i,j] = v/b
        
#%%


fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), facecolor="white")
ax1, ax2 = axs
ax1.imshow(mat_floor2, cmap="Blues", vmin=0, vmax=1)
ax2.imshow(mat_cat2, cmap="Blues", vmin=0, vmax=1)

ticklabels = [floor_order2, cat_order]
diags = [diag_floor, diag_cat]
mats = [mat_floor, mat_cat]
mats2 = [mat_floor2, mat_cat2]

ax1.set_title("(a) Between floor groups", loc="left")
ax2.set_title("(b) Between location categories", loc="left")

for i, ax in enumerate(axs):
    tlab = ticklabels[i]
    ax.set_xticks(list(range(len(tlab))))
    ax.set_yticks(list(range(len(tlab))))
    ax.set_xticklabels(tlab, rotation=35, ha="right")
    ax.set_yticklabels(tlab)
    diag = diags[i]
    for j,v in enumerate(diag):
        ax.text(j,j,int(v), ha="center", va='center', zorder=5, c='k', style='normal', fontweight='bold')
    
    mat = mats2[i]
    for a in range(len(diag)):
        for b in range(len(diag)):
            if a==b:continue
            v = mat[a,b]
            if v==0:continue
            fc = 'k' if round(v,2)<0.65 else 'w'
            ax.text(b,a,"{:.2f}".format(v), ha="center", va='center', zorder=5, c=fc, fontsize=10)
    
plt.tight_layout()
plt.savefig("cross_groups_cats_inoutloop_mat_norm_rowsum_2.png", dpi=300, bbox_inches="tight")

#%%
rowsum = [ v[0] for v in mat_floor.sum(axis=1).tolist() ]
colsum = mat_floor.sum(axis=0).tolist()[0]
#print(rowsum, colsum)
mat_floor2 = np.matrix(np.zeros(mat_floor.shape))
mat_floor3 = np.matrix(np.zeros(mat_floor.shape))
for i in range(mat_floor.shape[0]):
    for j in range(mat_floor.shape[1]):
        v = mat_floor[i,j]
        b = rowsum[i]
        mat_floor2[i,j] = v/b
        c = colsum[j]
        #print(v,b,c)
        mat_floor3[i,j] = v/c
"""
made changes -- removed row2 and used row -- takes all connections to calculate entropy
"""
floor_out_entropy = []
floor_in_entropy = []
for i in range(mat_floor2.shape[0]):
    row = mat_floor2[i,:].tolist()[0]
    #row2 = [ r for r in row if r>0 ]
    #print(row2)
    H = -sum([ p*np.log2(p) if p>0 else 0. for p in row ])/np.log2(len(row)) if len(row)>1 else 0
    floor_out_entropy.append(H)
    col = [ v[0] for v in mat_floor3[:,i].tolist() ]
    #col2 = [ r for r in col if r>0 ]
    H2 = -sum([ p*np.log2(p) if p>0 else 0. for p in col ])/np.log2(len(col)) if len(col)>1 else 0
    floor_in_entropy.append(H2)
print([ "{:.3f}".format(v) for v in floor_out_entropy ])
print([ "{:.3f}".format(v) for v in floor_in_entropy ])

rowsum = [ v[0] for v in mat_cat.sum(axis=1).tolist() ]
colsum = mat_cat.sum(axis=0).tolist()[0]
mat_cat2 = np.matrix(np.zeros(mat_cat.shape))
mat_cat3 = np.matrix(np.zeros(mat_cat.shape))
for i in range(mat_cat.shape[0]):
    for j in range(mat_cat.shape[1]):
        v = mat_cat[i,j]
        b = rowsum[i]
        mat_cat2[i,j] = v/b
        c = colsum[j]
        mat_cat3[i,j] = v/c
        
cat_out_entropy = []
cat_in_entropy = []
for i in range(mat_cat2.shape[0]):
    row = mat_cat2[i,:].tolist()[0]
    H = -sum([ p*np.log2(p) if p>0 else 0. for p in row ])/np.log2(len(row))
    H = 0. if H==-0. else H
    cat_out_entropy.append(H)
    col = [ v[0] for v in mat_cat3[:,i].tolist() ]
    H2 = -sum([ p*np.log2(p) if p>0 else 0. for p in col ])/np.log2(len(col))
    H2 = 0. if H2==-0. else H2
    cat_in_entropy.append(H2)
print([ "{:.3f}".format(v) for v in cat_out_entropy ])
print([ "{:.3f}".format(v) for v in cat_in_entropy ])
#%%
df_entropy_floor = pd.DataFrame.from_dict({"Floor_group":floor_order+["",""],
                                           "floor_out":[ "{:.3f}".format(round(v,3)) for v in floor_out_entropy ]+["",""], 
                                           "floor_in":[ "{:.3f}".format(round(v,3)) for v in floor_in_entropy ]+["",""],
                                          })
df_entropy_cat = pd.DataFrame.from_dict({"Location_category":cat_order,
                                         "cat_out":[ "{:.3f}".format(round(v,3)) for v in cat_out_entropy ],
                                         "cat_in":[ "{:.3f}".format(round(v,3)) for v in cat_in_entropy ], 
                                        })
df_entropy_hor = df_entropy_floor.join(df_entropy_cat, )
#print(df_entropy_floor.to_markdown())
#print(df_entropy_cat.to_markdown())
#print(df_entropy_hor.to_markdown())


























