import os
import sys

script=os. getcwd()
# paths=script.split("/")
#base_path='/'.join([paths[i] for i in range(paths.index("GSSLEmbedding"))])+"/GSSLEmbedding/"
base_path=script
print(base_path)

dataset_info_local={
    "test"    :{"name":"test",
                "path":"/Users/siddharthashankardas/Purdue/Dataset/test_data/",
                "output_path":"/Users/siddharthashankardas/Purdue/Dataset/test_data/"},

    "karate"    :{"name":"karate",
                  "path":"/"},
    "yelp"      :{"name":"yelp",
                  "path":"/Users/siddharthashankardas/Purdue/Dataset/Yelp/",
                  "output_path":"/Users/siddharthashankardas/Purdue/Dataset/Yelp/"},
    "dbpedia"   :{"name":"dbpedia",
                  "path":"/Users/siddharthashankardas/Purdue/Dataset/DBpedia/dbpedia_csv/",
                  "output_path":"/Users/siddharthashankardas/Purdue/Dataset/DBpedia/dbpedia_csv/"},
    "amazon"    :{"name":"amazon",
                  "path":"/Users/siddharthashankardas/Purdue/Dataset/AmazonReview/",
                  "output_path":"/Users/siddharthashankardas/Purdue/Dataset/AmazonReview/"},
    "imdb"      :{"name":"imdb",
                  "path":"/Users/siddharthashankardas/Purdue/Dataset/Imdb/aclImdb/",
                  "output_path":"/Users/siddharthashankardas/Purdue/Dataset/Imdb/aclImdb/"}
}

dataset_info_gilbreth={
    "yelp"      :{"name":"yelp",
                  "path":"/scratch/gilbreth/das90/Dataset/Yelp/",
                  "output_path":"/scratch/gilbreth/das90/Dataset/Yelp/"},
    "dbpedia"   :{"name":"dbpedia",
                  "path":"/scratch/gilbreth/das90/Dataset/DBpedia/dbpedia_csv/",
                  "output_path":"/scratch/gilbreth/das90/Dataset/DBpedia/dbpedia_csv/"},
    "imdb"      :{"name":"imdb",
                  "path":"/scratch/gilbreth/das90/Dataset/Imdb/aclImdb/",
                  "output_path":"/scratch/gilbreth/das90/Dataset/Imdb/aclImdb/"}
}

pretrained_model_local={
    "GOOGLE"        :{"name":"GOOGLE",
                      "path":"/Users/siddharthashankardas/Purdue/Dataset/Model/word2vec/GoogleNews-vectors-negative300.bin"},

    "GLOVE"         :{"name":"GLOVE",
                      "path":"/Users/siddharthashankardas/Purdue/Dataset/Model/glove.6B/gensim_glove.6B.300d.txt"},

    "CYBERSECURITY" :{"name":"CYBERSECURITY",
                      "path":"/Users/siddharthashankardas/Purdue/Dataset/Model/cybersecurity/1million.word2vec.model"}
}

pretrained_model_gilbreth={
    "GOOGLE"        :{"name":"GOOGLE",
                      "path":"/scratch/gilbreth/das90/Dataset/Model/word2vec/GoogleNews-vectors-negative300.bin"},

    "GLOVE"         :{"name":"GLOVE",
                      "path":"/scratch/gilbreth/das90/Dataset/Model/glove.6B/gensim_glove.6B.300d.txt"},

    "CYBERSECURITY" :{"name":"CYBERSECURITY",
                      "path":"/scratch/gilbreth/das90/Dataset/Model/cybersecurity/1million.word2vec.model"}
}


dataset_info={}
pretrained_model={}

print(os.uname())
pc_name=(os.uname()[1]).split('-')[0]
print("Pc-Name: ",pc_name)

if(pc_name=="Siddharthas"):
    dataset_info=dataset_info_local
    pretrained_model=pretrained_model_local
elif(pc_name == "gilbreth"):
    dataset_info=dataset_info_gilbreth
    pretrained_model = pretrained_model_gilbreth
else:
    sys.exit(0)

executable=''

if(pc_name=="Siddharthas"):
    executable=base_path+'/GraphConstruction/Bmatch/Release_osx/BMatchingSolver'
elif(pc_name == "gilbreth"):
    executable=base_path+'/GraphConstruction/Bmatch/Release_linux/BMatchingSolver'
else:
    print(pc_name," not matched")
    sys.exit(0)