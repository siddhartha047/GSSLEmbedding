import os
import sys
import gensim

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
    "reuters10w2v":{
        "name":"reuters10w2v",
        "input_path":"",
        "output_path":data_path+"Reuters10w2v/"
    },
    "reuters10tfidf":{
        "name":"reuters10_tfidf",
        "input_path":"",
        "output_path":data_path+"Reuters10tfidf/"
    },
    "newsgroup20w2v": {
        "name": "newsgroup",
        "input_path": "",
        "output_path": data_path + "Newsgroup20w2v/"
    },
    "newsgroup20tfidf": {
            "name": "newsgroup20_tfidf",
            "input_path": "",
            "output_path": data_path + "Newsgroup20tfidf/"
        },
    "imdb8w2v": {
        "name": "imdb",
        "input_path": data_path + "Imdb/aclImdb/",
        "output_path": data_path + "Imdb8w2v/"
    },
    "imdb8tfidf": {
        "name": "imdb",
        "input_path": data_path + "Imdb/aclImdb/",
        "output_path": data_path + "Imdb8tfidf/"
    },
    "dbpedia14w2v": {
        "name": "dbpedia",
        "input_path": data_path + "DBpedia/dbpedia_csv/",
        "output_path": data_path + "DBpedia14w2v/"
    },
    "dbpedia14tfidf": {
            "name": "dbpedia",
            "input_path": data_path + "DBpedia/dbpedia_csv/",
            "output_path": data_path + "DBpedia14tfidf/"
    },
    "yelp4w2v": {
            "name": "yelp",
            "input_path": data_path + "Yelp/",
            "output_path": data_path + "Yelp4w2v/"
    },
    "yelp4tfidf": {
            "name": "yelp",
            "input_path": data_path + "Yelp/",
            "output_path": data_path + "Yelp4tfidf/"
    },
    "amazon4tfidf": {
            "name": "amazon",
            "input_path": data_path + "AmazonReview/",
            "output_path": data_path + "Amazon4tfidf/"
    }

}

def load_model(model_name):
    if (model_name == "GLOVE"):
        model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(pretrained_model[model_name]["path"]), binary=False,
                                                                encoding="ISO-8859-1")
    elif (model_name == "GOOGLE"):
        model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(pretrained_model[model_name]["path"]), binary=True)

    else:
        print("Model not implemented yet")
        sys.exit(0)

    return model


script=os. getcwd()
base_path=script
print(base_path)

executable=''

if(pc_name=="Siddharthas" or sysname=="Darwin"):
    executable=base_path+'/Release_osx/BMatchingSolver'
elif(pc_name in ["dhaka","gilbreth", "rice", "snyder","scholar","halstead"]):
    executable=base_path+'/DatasetProcessing/Release_linux/BMatchingSolver'
else:
    print(pc_name," not matched")
    sys.exit(0)