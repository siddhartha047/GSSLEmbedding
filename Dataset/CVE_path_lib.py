import sys
import os

base_dir='/Users/sid/Purdue/Research/PNNLSummer2019/Codes/GNN_Data/CVE/'
cve_result_dir='/Users/sid/Purdue/Research/PNNLSummer2019/GNN_Results/'

model_base_dir=base_dir+'model/'
temp_dir=base_dir

print(os.uname())

if os.uname()[0].find('Darwin')!=-1:
    base_dir = '/Users/sid/Purdue/Research/PNNLSummer2019/Codes/GNN_Data/CVE/'
    cve_result_dir = '/Users/sid/Purdue/Research/PNNLSummer2019/GNN_Results/'

    model_base_dir = base_dir + 'model/'
    temp_dir = base_dir


elif os.uname()[1].find('purdue')!=-1:
    base_dir = '/home/das90/GNNcodes/GNN_Data/CVE/'
    cve_result_dir = '/home/das90/GNNcodes/GNN_Results/'
    model_base_dir = '/scratch/gilbreth/das90/GNN_data/model/'
    temp_dir='/scratch/gilbreth/das90/GNN_data/'

# elif os.uname()[1].find('pnl.gov')!=-1:
#     base_dir = '/people/dass548/sid/GNNcodes/GNN_Data/CVE/'
#     cve_result_dir = '/people/dass548/sid/GNNcodes/GNN_Results/'
#     model_base_dir = base_dir + 'model/'


cve_dir = base_dir+'cve_cwe/'


model_path = model_base_dir+'1million.word2vec.model'
model_file1 = model_base_dir+'1million.word2vec.model.syn1neg.npy'
model_file2 = model_base_dir+'1million.word2vec.model.wv.syn0.npy'
GLOVE_DIR = model_base_dir+'/glove.6B/glove.6B.100d.txt'
GOOGLE_NEWS = model_base_dir+'/word2vec/GoogleNews-vectors-negative300.bin'


#cve_data = cve_dir+'CVE_allitems_280619_comma.csv'
cve_data = cve_dir+'CVE_allitems_280619_using'

cwe_dir = cve_dir
cwe_data = cve_dir+'CVE2CWE.txt'
cwe_rc = cve_dir+'CWE_RC_1000.csv'
cwe_dc = cve_dir+'CWE_DC_699.csv'
cwe_ac = cve_dir+'CWE_AC_1008.csv'

cve_labeled = cve_dir+'CVE_CWE_labeled_items'
cve_all = cve_dir+'CVE_CWE_all_items'
stats_file = cve_dir+'stats.pkl'
cwe_map_file = cve_dir+'cwe_map.pkl'
index_cwe_map_file = cve_dir+'index_cwe_map.pkl'

use_labeled_only=True

if use_labeled_only:
    cve_file=cve_labeled
    cve_graph_dir = base_dir+'cve_graph_labeled/'
else:
    cve_file=cve_all
    cve_graph_dir = base_dir+'cve_graph_all/'

if os.uname()[0].find('Darwin')!=-1:
    pass

elif os.uname()[1].find('purdue')!=-1:
    if(use_labeled_only==False):
        cve_graph_dir ='/scratch/gilbreth/das90/GNN_data/cve_graph_all/'

cve_graph=cve_graph_dir+'CVE_CWE_graph.npz'
cve_x=cve_graph_dir+'features'
cwe_cluster_number=cve_graph_dir+'cwe_cluster'
cve_text=cve_graph_dir+'descriptions'
cve_y=cve_graph_dir+'labels'
cve_gephi=cve_graph_dir+"CVE_CWE.gexf"

cve_label_mask=cve_graph_dir+'label_mask'
cve_train_index=cve_graph_dir+'train_index'
cve_test_index=cve_graph_dir+'test_index'
cve_val_index=cve_graph_dir+'val_index'



