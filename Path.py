import os
import sys

local_path="/Users/siddharthashankardas/Purdue/Dataset/"
gilbreth_scratch_path="/scratch/gilbreth/das90/Dataset/"

#dataset location in the system
dataset_info_local={
    "test"    :{"name":"test",
                "path":local_path+"test_data/",
                "output_path":local_path+"test_data/"},

    "karate"    :{"name":"karate",
                    "path":local_path+"Karate/",
                    "output_path":local_path+"Karate/"},

    "reuters"    :{"name":"reuters",
                "path":"",
                "output_path":local_path+"Reuters/"},

    "newsgroup": {"name": "newsgroup",
                "path": "",
                "output_path": local_path + "Newsgroup/"},

    "imdb"      :{"name":"imdb",
                  "path":local_path+"Imdb/aclImdb/",
                  "output_path":local_path+"Imdb/aclImdb/"},

    "dbpedia"      :{"name":"dbpedia",
                  "path":local_path+"DBpedia/dbpedia_csv/",
                  "output_path":local_path+"DBpedia/"},

    "Yelp"      :{"name":"yelp",
                  "path":local_path+"Yelp/",
                  "output_path":local_path+"Yelp/"}
}

dataset_info_gilbreth={
    "reuters"    :{"name":"reuters",
                "path":"",
                "output_path":gilbreth_scratch_path+"Reuters/"},
    "newsgroup": {"name": "newsgroup",
                  "path": "",
                  "output_path": gilbreth_scratch_path + "Newsgroup/"},
    "imdb"      :{"name":"imdb",
                  "path":gilbreth_scratch_path+"Imdb/aclImdb/",
                  "output_path":gilbreth_scratch_path+"Imdb/aclImdb/"}
}

#pretrained model location
pretrained_model_local={
    "GOOGLE"        :{"name":"GOOGLE",
                      "path":local_path+"/Model/word2vec/GoogleNews-vectors-negative300.bin"},

    "GLOVE"         :{"name":"GLOVE",
                      "path":local_path+"/Model/glove.6B/gensim_glove.6B.300d.txt"},

    "CYBERSECURITY" :{"name":"CYBERSECURITY",
                      "path":local_path+"Model/cybersecurity/1million.word2vec.model"}
}

pretrained_model_gilbreth={
    "GOOGLE"        :{"name":"GOOGLE",
                      "path":gilbreth_scratch_path+"Model/word2vec/GoogleNews-vectors-negative300.bin"},

    "GLOVE"         :{"name":"GLOVE",
                      "path":gilbreth_scratch_path+"Model/glove.6B/gensim_glove.6B.300d.txt"},

    "CYBERSECURITY" :{"name":"CYBERSECURITY",
                      "path":gilbreth_scratch_path+"Model/cybersecurity/1million.word2vec.model"}
}

dataset_info={}
pretrained_model={}

print(os.uname())
pc_name=(os.uname()[1]).split('-')[0]
sysname=(os.uname()[0]).split('-')[0]
print("Pc-Name: ",pc_name)
print("Sysname: ",sysname)


if(pc_name=="Siddharthas" or sysname=="Darwin"):
    dataset_info=dataset_info_local
    pretrained_model=pretrained_model_local
elif(pc_name == "gilbreth"):
    dataset_info=dataset_info_gilbreth
    pretrained_model = pretrained_model_gilbreth
else:
    print(pc_name, " not matched, please fix pc name in the path file")
    sys.exit(0)

script=os. getcwd()
base_path=script
print(base_path)
executable=''

if(pc_name=="Siddharthas" or sysname=="Darwin"):
    executable=base_path+'/GraphConstruction/Bmatch/Release_osx/BMatchingSolver'
elif(pc_name in ["gilbreth", "rice", "snyder","scholar","halstead"]):
    executable=base_path+'/GraphConstruction/Bmatch/Release_linux/BMatchingSolver'
else:
    print("Warning::: ",pc_name," not matched, please fix pc name in the path file")