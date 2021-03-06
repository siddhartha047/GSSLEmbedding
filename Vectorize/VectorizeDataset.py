import timeit
import os
import sys

def main(dataset_info,config,method_config):

    dataset_name=dataset_info['name']
    home_dir=dataset_info['path']
    data_out_dir=dataset_info['output_path']
    output_dir=dataset_info['output_path']+config['method']+'/'
    if not os.path.exists(output_dir):
        print("Creating directory: ",output_dir)
        os.makedirs(output_dir)

    print("Data directory: ",home_dir)
    print("Output directory: ",output_dir)

    output_file = output_dir + dataset_name+'_review.mtx'
    output_label = output_dir + dataset_name+'_review.label'
    output_data = output_dir + dataset_name+'_data'
    model_file = output_dir + dataset_name+'.model'

    data=[]
    data_rating=[]
    data_vector=[]

    from Dataset.Dataset_lib import read_dataset
    (data, data_rating)=read_dataset(dataset_name,home_dir,data_out_dir,config['load_saved'])

    import numpy as np
    start_reading = timeit.default_timer()
    data = np.array(data)
    data_rating = np.array(data_rating)
    stop_reading = timeit.default_timer()
    print('Time to convert into numpy: ', stop_reading - start_reading)


    print(config['method'], " Training started")
    if(config['method']=="doc2vec"):
        from Vectorize.Doc2Vec.Doc2VecTrain import get_vector
        max_epochs = method_config['max_epochs']
        vec_size = method_config['vec_size']
        data_vector=get_vector(data,model_name=model_file,max_epochs=max_epochs,vec_size=vec_size,use_saved=config['load_saved'],visualize=config['visualize'])

    elif(config['method']=="word2vec"):
        from Vectorize.Word2Vec.Word2Vec import learn
        pretrained_model=method_config['pretrained_model']
        vec_size = method_config['vec_size']
        model_name = output_dir +method_config['pretrained_model_name']+ 'w2v.model'
        model_info=pretrained_model[method_config['pretrained_model_name']]
        data_doc=[item.split() for item in data]
        data_vector=learn(data_doc,model_info,vec_size=vec_size,model_name=model_name,load_saved=config['load_saved'],visualize=config['visualize'])

    elif(config['method']=="word2vec_avg"):
        from Vectorize.Word2VecAvg.Word2Vec_avg import learn
        pretrained_model = method_config['pretrained_model']
        vec_size = method_config['vec_size']
        model_name = output_dir +method_config['pretrained_model_name']+ 'w2v.model'
        model_info = pretrained_model[method_config['pretrained_model_name']]
        data_doc = [item.split() for item in data]
        data_vector=learn(data_doc, model_info, vec_size=vec_size, visualize=config['visualize'])

    elif (config['method'] == "TF_IDF"):
        from Vectorize.Transformer.tfidf import tf_idf
        data_vector=tf_idf(data,method_config)
    elif (config['method'] == "LSI"):
        from Vectorize.Transformer.lsi import lsi
        data_vector=lsi(data,method_config)
    elif (config['method'] == "LDA"):
        from Vectorize.Transformer.lda import lda
        data_vector=lda(data,method_config)
    else:
        sys.exit("not implemented yet")
    print(config['method'], " Training ended")

    print(data.shape)
    print(data_rating.shape)
    print(data_vector.shape)

    print("Saving data in ", config["saving_format"], "format")

    if("numpy" in config["saving_format"]):
        from Dataset.Lib import save_vector_rating_numpy,save_vector_scipy

        if (type(data_vector).__name__ == "csr_matrix"):

            save_vector_scipy(output_dir,data_vector)

            print("Converting scipy sparse to dense,")
            data_vector_dense=data_vector.todense()
            save_vector_rating_numpy(output_dir,data_vector_dense, data_rating)
        else:
            save_vector_rating_numpy(output_dir,data_vector,data_rating)

    if("mtx2" in config["saving_format"]):
        from Dataset.Lib import save_vector_mtx2
        save_vector_mtx2(output_dir, dataset_name, data_vector)

        return

    if ("mtx" in config["saving_format"]):
        from Dataset.Lib import save_data
        save_data(data_vector, data_rating, output_file, output_label, output_data, comment=dataset_name)

    if ("mat" in config["saving_format"]):
        from Dataset.Lib import save_data_mat
        save_data_mat(output_dir,data_vector,data_rating)

    if ("txt" in config["saving_format"]):
        from Dataset.Lib import save_data_txt
        if (type(data_vector).__name__ == "csr_matrix"):
            print("Converting scipy sparse to dense,")
            data_vector_dense=data_vector.todense()
            save_data_txt(output_dir, data_vector_dense, data_rating)

        else:
            save_data_txt(output_dir, data_vector, data_rating)

    print("Data saving done")


if __name__ == '__main__':
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print('Time: ', stop - start)