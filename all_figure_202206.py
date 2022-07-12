import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

def get_data():
    dg = nx.read_gexf("../data/ka_network_graph_a_17mar2021_with_mobility.gexf")
    print('Number of nodes: ',len(dg.nodes()))
    print('Number of edges: ',len(dg.edges()))

    with open('pos_files/pos_all.json', 'r') as fp:
        pos = json.load(fp)

    node_levels = defaultdict(list)
    for n,d in dg.nodes(data=True):
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
    node_group = {}
    for k,v in groups.items():
        #print(k,v)
        for n in v:
            node_group[n] = k

    return dg, pos, node_group, groups


def get_node_colors(dg, node_group):
    clrs = sns.color_palette("tab10")
    # sns.palplot(clrs)
    #group_levels = ['carparks', 'L1', 'L2', 'L6', 'rooftop garden', 'residential']
    group_levels = [ 'residential', 'L1', 'rooftop_garden', 'L2', 'carparks', 'L6' ]
    group_colors = { group_levels[i]: clrs[i] for i in range(len(group_levels))}
    group_colors["L7"] = clrs[6]
    group_colors["L3-L4"] = clrs[6]
    group_colors["crossing"] = clrs[7]
    nodelist = list(dg.nodes())
    loc_listb = group_levels
    loc_listb2 = [ a.replace("_", " ").capitalize() for a in loc_listb ] + ["Other"]
    #loc_clrsb = { loc_listb[i]:clrs[i] for i in range(len(loc_listb)) }
    n_clrsb = [ group_colors[node_group[n]] for n in nodelist ]
    return clrs, n_clrsb, group_colors

def draw_fig2_network(dg, pos, clrs, n_clrsb, group_colors):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="w")
    nodelist = list(dg.nodes())
    node_incoming = dg.in_degree(weight="mobility_number")
    nsize = [ (node_incoming[n]+1)*2 for n in nodelist ]
    nx.draw_networkx_nodes(dg, nodelist=nodelist, pos=pos, ax=ax, node_size=nsize, node_color=n_clrsb)
    #nx.draw_networkx_edges(dg, pos=pos, ax=ax, )
    edgelist = dg.edges()
    #print(edgelist)
    edgeweight = [ dg[u][v]["mobility_number"] for u,v in edgelist ]
    edgewidth = [ np.log(w)*.75+1 if w>0 else 1 for w in edgeweight ]
    nx.draw_networkx_edges(dg, pos=pos, ax=ax, edgelist=edgelist, width=edgewidth,
                           connectionstyle="arc3,rad=0.1", edge_color="xkcd:navy", alpha=.6)

    clabels = [ 'carparks', 'L1', 'L2', 'L6', 'rooftop_garden',  'residential']#, "Other" ]#loc_listb2
    clabels2 = [ 'Car-parks', 'Level-1', 'Level-2', 'Level-6', 'Rooftop garden',  'Residential']
    clrs2 = [ group_colors[k] for k in clabels ]
    legend_elements = []
    for l,c in zip(clabels2, clrs2):
        ll = l.replace("_", " ").capitalize()
        l2d = Line2D([0], [0], color=c, lw=0, marker="o", ls="", label=ll)
        legend_elements.append(l2d)
    l2do = Line2D([0], [0], color=clrs[6], lw=0, marker="o", ls="", label="Other")
    legend_elements.append(l2do)
    ax.legend(handles=legend_elements, ncol=2, loc='lower right')
    ax.set_xlim([-0.8, 0.7])
    ax.set_ylim([-1.1, 0.8])
    plt.tight_layout()
    # ax.set_title('(a) Mobility graph', loc="left")
    fig.savefig(os.path.join("figs_202206", "fig2.png"), dpi=300, bbox_inches="tight")
    #plt.show()

def get_group_count(dg, groups):
    group_levels = ['carparks', 'L1', 'L2', 'L6', 'rooftop_garden', 'residential']

    group_count = {}
    #loc_listb = ['Community_Street', 'Garden_Street', 'Residential_Street', 'Commercial_Street', 'Social_Space', 'Corridor', 'Vertical_Street', 'Entrance_Street']
    loc_listb = ['Residential_Street', 'Social_Space', 'Garden_Street', 'Commercial_Street', 'Entrance_Street', 'Community_Street', 'Vertical_Street', 'Corridor']

    for i,k in enumerate(group_levels):
        #temp_pos = pos_by_floor[k]
        ns = groups[k]
        kg = dg.subgraph(ns)
        edgelist = kg.edges()
        edgeweight = [ kg[u][v]["mobility_number"] for u,v in edgelist ]
        destinations = [ v for u,v in edgelist ]
        dest_cat = [ dg.nodes[n]["location_type"] for n in destinations ]
        temp = defaultdict(int)
        for c,v in zip(dest_cat, edgeweight):
            temp[c]+=v
        temp2 = [ temp[c] for c in loc_listb ]
        group_count[k] = temp2
        #break
    oth = [ k for k in groups.keys() if not(k in group_levels) ]
    temp = defaultdict(int)
    for k in oth:
        #print(k)
        ns = groups[k]
        kg = dg.subgraph(ns)
        edgelist = kg.edges()
        edgeweight = [ kg[u][v]["mobility_number"] for u,v in edgelist ]
        destinations = [ v for u,v in edgelist ]
        dest_cat = [ dg.nodes[n]["location_type"] for n in destinations ]
        #temp = defaultdict(int)
        for c,v in zip(dest_cat, edgeweight):
            temp[c]+=v
    temp2 = [ temp[c] for c in loc_listb ]
    group_count["other"] = temp2
    #print(group_count)
    df_group_count = pd.DataFrame.from_dict(group_count, orient='index', columns=loc_listb)
    loc_listb2 = [ a.split("_")[0].lower() if not(a=="Corridor") else "corridor" for a in loc_listb ]

    for i in range(len(loc_listb2)):
        df_group_count[loc_listb2[i]] = df_group_count.iloc[:,:i+1].sum(axis=1)
    df_group_count
    return df_group_count, group_count, loc_listb

def draw_fig1_barh(dg, groups, clrs):
    df_group_count, group_count, loc_listb = get_group_count(dg, groups)
    loc_listb2 = [ a.split("_")[0].lower() if not(a=="Corridor") else "corridor" for a in loc_listb ]
    loc_clrsb = { loc_listb2[i]:clrs[i] for i in range(len(loc_listb)) }

    loc_listb = ['Residential_Street', 'Social_Space', 'Garden_Street', 'Commercial_Street', 'Entrance_Street', 'Community_Street', 'Vertical_Street', 'Corridor']
    loc_listb2 = [ a.split("_")[0].lower() for a in loc_listb ]
    n_clrs_cat = [ clrs[i] for i in range(len(loc_listb2)) ]
    loc_clrsb = {k:v for k,v in zip(loc_listb2, clrs)}

    group_levels_label = ['Car-parks', 'Level-1', 'Level-2', 'Level-6', 'Rooftop garden', 'Residential', 'Other']

    fig, ax = plt.subplots(1, 1, figsize=(6,6), facecolor="w")

    for i,c in enumerate(loc_listb2):
        vs = df_group_count[c].tolist()
        ys = list(range(len(vs)))
        #c2 = loc_listb2[i]
        ax.barh(ys, vs, fc=loc_clrsb[c], label=c.capitalize(), zorder=len(loc_listb2)-i+2, alpha=.9)
    #ax.set_title("({}) {}".format(labs[i], k.capitalize()), loc="left")
    ax.legend(ncol=3, loc='upper right')
    ax.set_yticks(list(range(len(group_count))))
    #ax.set_yticklabels([y.capitalize().replace("_", " ") for y in list(group_count.keys())])
    ax.set_yticklabels(group_levels_label)
    ax.set_xlabel("Incoming flow")
    #ax.set_xscale("log")
    plt.tight_layout()
    fig.savefig(os.path.join("figs_202206", "fig1.png"), dpi=300, bbox_inches="tight")
    #plt.show()


def get_clr_fig3():
    clrs = sns.color_palette("tab10")
    loc_listb = ['Residential_Street', 'Social_Space', 'Garden_Street', 'Commercial_Street', 'Entrance_Street', 'Community_Street', 'Vertical_Street', 'Corridor']
    loc_listb2 = [ a.split("_")[0] for a in loc_listb ]
    loc_clrsb = {k:v for k,v in zip(loc_listb, clrs)}
    #n_clrs_cat = [ clrs[i] for i in range(len(loc_listb2)) ]
    return loc_clrsb, loc_listb

def draw_fig3_networks(dg, groups):
    with open('pos_files/pos_all_by_floor_w_ext.json', 'r') as fp:
        pos_by_floor_ext = json.load(fp)
    groups_with_ext = {}
    for k,ns in groups.items():
        #print(len(ns))
        ns2 = []#ns.copy()
        for n in ns:
            temp_ns = list(dg.predecessors(n))
            ns2.extend(temp_ns)
            temp_ns = list(dg.successors(n))
            ns2.extend(temp_ns)
            #break
        ns2 = ns+ns2
        ns2 = list(set(ns2))
        #print(len(ns2))
        groups_with_ext[k] = ns2
        #break
    loc_clrsb, loc_listb = get_clr_fig3()
    group_levels = ['carparks', 'L1', 'L2', 'L6', 'rooftop_garden', 'residential']
    group_levels_label = ['Car-parks', 'Level-1', 'Level-2', 'Level-6', 'Rooftop garden', 'Residential']


    fig, axg = plt.subplots(2, 3, figsize=(12,8), facecolor="w")
    axs = axg.flatten()
    labs = "abcdef"
    for i,k in enumerate(group_levels):
        k2 = group_levels_label[i]
        ax = axs[i]
        temp_pos = pos_by_floor_ext[k]
        ns = groups[k]
        ns2 = groups_with_ext[k]
        ext_ns = list(set(ns2) - set(ns))

        kg = dg.subgraph(ns)
        kg2 = dg.subgraph(ns2)
        n_clrs_temp = [ loc_clrsb[dg.nodes[n]["location_type"]] for n in ns ]
        nx.draw_networkx_nodes(kg2, nodelist=ns, pos=temp_pos, ax=ax, node_size=35, node_color=n_clrs_temp)
        #nx.draw_networkx_nodes(kg2, nodelist=ext_ns, pos=temp_pos, ax=ax, node_size=1, node_color='lightgray')
        edgelist = kg.edges()
        edgeweight = [ kg[u][v]["mobility_number"] for u,v in edgelist ]
        edgewidth = [ (np.log2(w)*1+1)/2 if w>0 else 1 for w in edgeweight ]
        nx.draw_networkx_edges(kg2, pos=temp_pos, ax=ax, edgelist=edgelist, width=edgewidth,
                               connectionstyle="arc3,rad=0.1", edge_color="xkcd:navy", alpha=.6)
        edgelist2 = (kg2.edges())
        edgelist_ext = []
        for u,v in edgelist2:
            if (u,v) in edgelist:
                continue
            if (u in ns) or (v in ns):
                edgelist_ext.append((u,v))
        nx.draw_networkx_edges(kg2, pos=temp_pos, ax=ax, edgelist=edgelist_ext, width=.5,
                               connectionstyle="arc3,rad=0.1", arrowsize=1, edge_color="gray", alpha=.8)
        #ax.text(0.01,0.99,k, ha="left", va="top", transform=ax.transAxes, )
        ax.set_title("({}) {}".format(labs[i], k2), loc="left")
        xlim_0 = ax.get_xlim()
        ylim_0 = ax.get_ylim()
        zoom = 0.85 if i<4 else 0.88
        xlim_1 = [ x*zoom for x in xlim_0 ]
        ylim_1 = [ y*zoom for y in ylim_0 ]
        ax.set_xlim(xlim_1)
        ax.set_ylim(ylim_1)

    clabels = loc_listb
    clrs = [ loc_clrsb[c] for c in clabels ]
    legend_elements = []
    for l,c in zip(clabels, clrs):
        ll = l.split("_")[0]
        l2d = Line2D([0], [0], color=c, lw=0, marker="o", ls="", label=ll)
        legend_elements.append(l2d)
    fig.legend(handles=legend_elements, ncol=8, loc='lower center', bbox_to_anchor=[.5, -.03])
    plt.tight_layout()
    fig.savefig(os.path.join("figs_202206", "fig3.png"), dpi=300, bbox_inches="tight")
    #plt.show()


def prepare_for_fig4(dg_floor, dg_cat, group_levels, loc_listb2):
    clrs = sns.color_palette("tab10")
    n_clrs_level = [ clrs[i] for i in range(len(group_levels)) ]
    n_clrs_cat = [ clrs[i] for i in range(len(loc_listb2)) ]
    nsize_floor = []
    for n in group_levels:
        edw = dg_floor[n][n]["mobility_number"]
        edw = edw+10
        nsize_floor.append(edw)
    nsize_cat = []
    for n in loc_listb2:
        if dg_cat.has_edge(n,n):
            edw = dg_cat[n][n]["mobility_number"]
        else:
            edw = 1
        edw = edw+10
        nsize_cat.append(edw)
    floor_lab2 = {
        "Carparks":"Car-parks",
        "L1":"Level-1",
        "L2":"Level-2",
        "L6":"Level-6",
        "Rooftop garden":"Rooftop garden",
        "Residential":"Residential",
    }
    with open('pos_files/pos_floor_group.json', 'r') as fp:
        pos_floor = json.load(fp)
    with open('pos_files/pos_cat.json', 'r') as fp:
        pos_cat = json.load(fp)
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


    return nsize_floor, nsize_cat, floor_lab2, pos_floor, pos_cat, pos_floor_lab, pos_cat_lab, n_clrs_level, n_clrs_cat

def make_dg_floor(dg, groups):
    group_levels = ['carparks', 'L1', 'L2', 'L6', 'rooftop_garden', 'residential', 'L3-L4', 'L7']
    groups2 = { k:groups[k] for k in group_levels }
    temp = []
    oth = [ k for k in groups.keys() if not(k in group_levels) ]
    for k in oth:
        ns = groups[k]
        temp.extend(ns)
    print(temp)
    groups2["other"] = temp
    floor_dic = {}
    for k,ns in groups2.items():
        k2 = k.replace("_", " ").capitalize()
        for n in ns:
            floor_dic[n] = k2
    group_levels = [ l.capitalize() for l in ['carparks', 'L1', 'L2', 'L6', 'rooftop garden', 'residential', 'L3-L4', 'L7'] ]
    dg_floor = nx.DiGraph()
    dg_floor.add_nodes_from(group_levels)
    for u,v,d in dg.edges(data=True):
        u2 = floor_dic[u]
        v2 = floor_dic[v]
        if d["mobility_number"]<=0: continue
        if dg_floor.has_edge(u2,v2):
            dg_floor[u2][v2]["mobility_number"]+=d["mobility_number"]
        else:
            dg_floor.add_edge(u2,v2, mobility_number=d["mobility_number"])
    group_levels = [ l.capitalize() for l in ['carparks', 'L1', 'L2', 'L6', 'rooftop garden', 'residential'] ]

    dg_floor2 = nx.DiGraph()
    dg_floor2.add_nodes_from(group_levels)
    check = ['L3-l4', 'L7']
    check_edge1 = {}
    check_edge2 = {}
    for u,v,d in dg_floor.edges(data=True):
        if not(u in check) and not(v in check):
            #print(u,v,d)
            dg_floor2.add_edge(u, v, mobility_number=d["mobility_number"])
        elif u=="L3-l4" or v=="L3-l4":
            check_edge1[(u,v)] = d["mobility_number"]
            #print(u,v,d)
        elif u=="L7" or v=="L7":
            check_edge2[(u,v)] = d["mobility_number"]
            print(u,v,d)
    print(dg_floor2.has_edge("L2", "L6"), dg_floor2.has_edge("L6", "L2"))
    six_to_two = min(check_edge1[("L6", "L3-l4")], check_edge1[("L3-l4", "L2")])
    print(six_to_two)
    print(dg_floor2["L6"]["L2"])
    dg_floor2["L6"]["L2"]["mobility_number"]+=six_to_two
    print(dg_floor2["L6"]["L2"])
    two_to_six = min(check_edge1[("L3-l4", "L6")], check_edge1[("L2", "L3-l4")])
    print(two_to_six)
    print(dg_floor2["L2"]["L6"])
    dg_floor2["L2"]["L6"]["mobility_number"]+=two_to_six
    print(dg_floor2["L2"]["L6"])
    print(dg_floor2.has_edge("Rooftop garden", "L6"), dg_floor2.has_edge("L6", "Rooftop garden"))
    dg_floor2.add_edge("Rooftop garden", "L6", mobility_number=0)
    dg_floor2.add_edge("L6", "Rooftop garden", mobility_number=0)
    six_to_top = min(check_edge2[("L6", "L7")], check_edge2[("L7", "Rooftop garden")])
    print(six_to_top)
    print(dg_floor2["L6"]["Rooftop garden"])
    dg_floor2["L6"]["Rooftop garden"]["mobility_number"]+=six_to_top
    print(dg_floor2["L6"]["Rooftop garden"])
    top_to_six = min(check_edge2[("L7", "L6")], check_edge2[("Rooftop garden", "L7")])
    print(top_to_six)
    print(dg_floor2["Rooftop garden"]["L6"])
    dg_floor2["Rooftop garden"]["L6"]["mobility_number"]+=top_to_six
    print(dg_floor2["Rooftop garden"]["L6"])
    dg_floor = dg_floor2
    return dg_floor

def make_dg_cat(dg):
    cat_dic = {}
    for n,d in dg.nodes(data=True):
        cat_dic[n] = d["location_type"].split("_")[0]
    loc_listb = ['Community_Street', 'Garden_Street', 'Residential_Street', 'Commercial_Street', 'Social_Space', 'Corridor', 'Vertical_Street', 'Entrance_Street']
    loc_listb2 = [ a.split("_")[0] for a in loc_listb ]
    dg_cat = nx.DiGraph()
    dg_cat.add_nodes_from(loc_listb2)
    for u,v,d in dg.edges(data=True):
        u2 = cat_dic[u]
        v2 = cat_dic[v]
        if d["mobility_number"]<=0: continue
        if dg_cat.has_edge(u2,v2):
            dg_cat[u2][v2]["mobility_number"]+=d["mobility_number"]
        else:
            dg_cat.add_edge(u2,v2, mobility_number=d["mobility_number"])
    return dg_cat

def draw_fig4_cat_network(dg, groups):
    dg_floor = make_dg_floor(dg, groups)
    dg_cat = make_dg_cat(dg)
    group_levels = [ l.capitalize() for l in ['residential', 'L1', 'rooftop garden', 'L2', 'carparks', 'L6' ]]#, 'other'] ]
    loc_listb = ['Residential_Street', 'Social_Space', 'Garden_Street', 'Commercial_Street', 'Entrance_Street', 'Community_Street', 'Vertical_Street', 'Corridor']
    loc_listb2 = [ a.split("_")[0] for a in loc_listb ]
    nsize_floor, nsize_cat, floor_lab2, pos_floor, pos_cat, pos_floor_lab, pos_cat_lab, n_clrs_level, n_clrs_cat = prepare_for_fig4(dg_floor, dg_cat, group_levels, loc_listb2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")
    ax1,ax2 = axs
    nx.draw_networkx_nodes(dg_floor, nodelist=group_levels, pos=pos_floor, ax=ax1, node_size=nsize_floor, node_color=n_clrs_level)
    edgelist = dg_floor.edges()
    edgelist = [(u, v) for u, v in edgelist if not(u==v)]
    edgeweight = [ dg_floor[u][v]["mobility_number"] for u,v in edgelist ]
    edgewidth = [ np.log(w)*1+1 if w>0 else 1 for w in edgeweight ]
    nx.draw_networkx_edges(dg_floor, pos=pos_floor, ax=ax1, edgelist=edgelist, width=edgewidth,
                           connectionstyle="arc3,rad=0.15", edge_color="xkcd:navy", alpha=.6, min_target_margin=15, min_source_margin=15)
    nx.draw_networkx_labels(dg_floor, pos=pos_floor_lab, labels=floor_lab2, ax=ax1)
    ax1.set_title("(a) Between floor groups", loc="left")

    nx.draw_networkx_nodes(dg_cat, nodelist=loc_listb2, pos=pos_cat, ax=ax2, node_size=nsize_cat, node_color=n_clrs_cat)
    edgelist = dg_cat.edges()
    edgelist = [(u, v) for u, v in edgelist if not(u==v)]
    edgeweight = [ dg_cat[u][v]["mobility_number"] for u,v in edgelist ]
    edgewidth = [ np.log(w)*1+1 if w>0 else 1 for w in edgeweight ]
    nx.draw_networkx_edges(dg_cat, pos=pos_cat, ax=ax2, edgelist=edgelist, width=edgewidth,
                           connectionstyle="arc3,rad=0.15", edge_color="xkcd:navy", alpha=.6, min_target_margin=15, min_source_margin=15)
    nx.draw_networkx_labels(dg_cat, pos=pos_cat_lab, ax=ax2)
    ax2.set_title("(b) Between location categories", loc="left")

    plt.tight_layout()
    plt.savefig(os.path.join("figs_202206", "fig4.png"), dpi=300, bbox_inches="tight")
    #plt.show()


def main():
    dg, pos, node_group, groups = get_data()
    clrs, n_clrsb, group_colors = get_node_colors(dg, node_group)
    draw_fig1_barh(dg, groups, clrs)
    draw_fig2_network(dg, pos, clrs, n_clrsb, group_colors)
    draw_fig3_networks(dg, groups)
    draw_fig4_cat_network(dg, groups)



if __name__ == '__main__':
    main()
