import os
import sys
print(os.uname())
pc_name=(os.uname()[1]).split('-')[0]
pc_name=pc_name.split('.')[0]
sysname=(os.uname()[0]).split('-')[0]
print("Pc-Name: ",pc_name)
print("Sysname: ",sysname)

local_path="/Users/siddharthashankardas/Purdue/Dataset/"
dhaka_path="/scratch/das90/Dataset/"

if(pc_name=="Siddharthas" or sysname=="darwin"):
    data_path=local_path
elif(pc_name=="dhaka"):
    data_path=dhaka_path
else:
    print("Data has not setup for this pc yet")
    sys.exit(0)

pretrained_model={
    "GOOGLE": {"name": "GOOGLE",
               "path": data_path+"Model/word2vec/GoogleNews-vectors-negative300.bin"},

    "GLOVE": {"name": "GLOVE",
              "path": data_path+"Model/glove.6B/gensim_glove.6B.300d.txt"}
}

dataset_path={
    "reuters10":{
        "name":"reuters10",
        "input_path":"",
        "output_path":data_path+"Reuters10/"
    },
    "newsgroup": {
        "name": "newsgroup",
        "input_path": "",
        "output_path": data_path + "NewsgroupX/"
    }
}