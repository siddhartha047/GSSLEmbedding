import sys

def read_dataset(dataset_name,home_dir,output_dir,load_saved):
    if (dataset_name == "imdb"):
        from Dataset import Imdb_read
        return Imdb_read.read(home_dir, output_dir,load_saved)

    elif (dataset_name == "dbpedia"):
        from Dataset import Dbpedia_read
        return Dbpedia_read.read(home_dir, output_dir,load_saved)

    else:
        print("unspecified dataset/not implemented")
        sys.exit(0)

