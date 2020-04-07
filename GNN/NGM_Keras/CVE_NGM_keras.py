import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Activation, Input,Dropout
from keras.models import Model
from GNN.NGM_Keras.NGM_Batch import batch_iter_test,batch_iter_train,batch_iter_val,batch_incremental
import timeit

import sys

###dataset
tf.reset_default_graph()
seed=1
#tf.random.set_random_seed(seed)

def getModel(config):
    # keras model

    input = Input(shape=(config['input_features'],))
    dense1 = Dense(config['hidden_neurons'][0], input_dim=config['input_features'])(input)
    act1 = Activation('relu')(dense1)

    if(config['dropout'][0]>0):
        drop1 = Dropout(config['dropout'][0])(act1)
        last = drop1
    else:
        last= act1

    for i in range(len(config['hidden_neurons']) - 1):
        dense = Dense(config['hidden_neurons'][i + 1])(last)
        act = Activation('relu')(dense)

        if (config['dropout'][i+1] > 0):
            drop1 = Dropout(config['dropout'][i+1])(act)
            last = drop1
        else:
            last= act

    dense3 = Dense(config['out_features'])(last)

    model = Model(inputs=input, outputs=dense3)

    print(model.summary())

    return model

def calculate_accuracy(sess,inputs,labels,test_corr_sum,test_pred_sum,batches):
    total_corr=0
    total_pred=0

    for t_batch in batches:
        tc,tp = sess.run([test_corr_sum,test_pred_sum], feed_dict={
            inputs: t_batch[0],
            labels: t_batch[1]
        })
        total_corr+=tc
        total_pred+=tp

    return total_corr/total_pred

def train(config,data):

    #global_step = tf.Variable(0, name='global_step', trainable=False) #dunno why yet

    alpha1 = tf.constant(1e-1, dtype=np.float32, name='a1')
    alpha2 = tf.constant(1e-1, dtype=np.float32, name='a2')
    alpha3 = tf.constant(1e-1, dtype=np.float32, name='a3')

    in_u1 = tf.placeholder(tf.float32, [None, config['input_features'], ], name="ull")
    in_v1 = tf.placeholder(tf.float32, [None, config['input_features'], ], name="vll")
    in_u2 = tf.placeholder(tf.float32, [None, config['input_features'], ], name="ulu")
    in_v2 = tf.placeholder(tf.float32, [None, config['input_features'], ], name="vlu")
    in_u3 = tf.placeholder(tf.float32, [None, config['input_features'], ], name="uuu")
    in_v3 = tf.placeholder(tf.float32, [None, config['input_features'], ], name="vuu")

    test_inputs = tf.placeholder(tf.float32, [None, config['input_features'], ], name="test_in")
    test_labels =  tf.placeholder(tf.float32, [None, config['out_features']], name="test_labels")

    labels_u1 = tf.placeholder(tf.float32, [None, config['out_features']], name="lull")
    labels_v1 = tf.placeholder(tf.float32, [None, config['out_features']], name="lvll")
    labels_u2 = tf.placeholder(tf.float32, [None, config['out_features']], name="lulu")

    cu1 = tf.placeholder(tf.float32, [None, ], name="Cull")
    cv1 = tf.placeholder(tf.float32, [None, ], name="Cvll")
    cu2 = tf.placeholder(tf.float32, [None, ], name="Culu")

    weights_ll = tf.placeholder(tf.float32, [None, ], name="wll")
    weights_lu = tf.placeholder(tf.float32, [None, ], name="wlu")
    weights_uu = tf.placeholder(tf.float32, [None, ], name="wuu")

    model=getModel(config)

    scores_u1 = model(in_u1)
    scores_v1 = model(in_v1)
    scores_u2 = model(in_u2)
    scores_v2 = model(in_v2)
    scores_u3 = model(in_u3)
    scores_v3 = model(in_v3)
    test_scores=model(test_inputs)

    #loss function ---------------
    part1 = tf.nn.softmax_cross_entropy_with_logits(logits=scores_u1, labels=tf.nn.softmax(scores_v1))
    part2 = tf.nn.softmax_cross_entropy_with_logits(logits=scores_v1, labels=tf.nn.softmax(scores_u1))

    part3 = tf.nn.softmax_cross_entropy_with_logits(logits=scores_u1, labels=labels_u1)
    part4 = tf.nn.softmax_cross_entropy_with_logits(logits=scores_v1, labels=labels_v1)

    part5 = tf.nn.softmax_cross_entropy_with_logits(logits=scores_u2, labels=tf.nn.softmax(scores_v2))
    part6 = tf.nn.softmax_cross_entropy_with_logits(logits=scores_v2, labels=tf.nn.softmax(scores_u2))

    part7 = tf.nn.softmax_cross_entropy_with_logits(logits=scores_u2, labels=labels_u2)

    # l1 = tf.reduce_sum(alpha1 * weights_ll * ((part1 + part2) / 2.0) + cu1 * part3 + cv1 * part4)
    # l2 = tf.reduce_sum(alpha2 * weights_lu * ((part5 + part6) / 2.0) + cu2 * part7)
    l1 = tf.reduce_mean(alpha1 * weights_ll * ((part1 + part2) / 2.0) + cu1 * part3 + cv1 * part4)
    l2 = tf.reduce_mean(alpha2 * weights_lu * ((part5 + part6) / 2.0) + cu2 * part7)

    part8 = tf.nn.softmax_cross_entropy_with_logits(logits=scores_u3, labels=tf.nn.softmax(scores_v3))
    part9 = tf.nn.softmax_cross_entropy_with_logits(logits=scores_v3, labels=tf.nn.softmax(scores_u3))

    #l3 = tf.reduce_sum(alpha3 * weights_uu * ((part8 + part9) / 2.0))
    l3 = tf.reduce_mean(alpha3 * weights_uu * ((part8 + part9) / 2.0))

    loss_function = l1 + l2 + l3
    #loss end ---------------

    #opt_op = tf.train.RMSPropOptimizer(0.5).minimize(loss_function)
    opt_op = tf.train.AdamOptimizer(0.001).minimize(loss_function)

    #performance ------------
    train_correct_prediction = tf.concat([
        tf.equal(tf.argmax(scores_u1, 1), tf.argmax(labels_u1, 1)),
        tf.equal(tf.argmax(scores_v1, 1), tf.argmax(labels_v1, 1)),
        tf.equal(tf.argmax(scores_u2, 1), tf.argmax(labels_u2, 1))
    ], axis=0)
    train_corr_sum = tf.reduce_sum(tf.cast(train_correct_prediction, "float"), name="train_accuracy")
    train_pred_num = tf.size(train_correct_prediction)

    test_correct_prediction = tf.equal(tf.argmax(test_scores, 1), tf.argmax(test_labels, 1))
    test_corr_sum = tf.reduce_sum(tf.cast(test_correct_prediction, "float"), name="test_accuracy")
    test_pred_num = tf.size(test_correct_prediction)

    #end performance

    init_op=tf.global_variables_initializer()

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    sess = tf.Session(config=session_conf)
    K.set_session(sess)

    sess.run(init_op)

    #epochs=config['epoch']
    epochs=12

    train_accuracies = list()
    val_accuracies = list()


    for epoch in range(epochs):

        print("======= EPOCH " + str(epoch + 1) + " ========")

        train_batches = batch_iter_train(data,config['batch_size'])
        #train_batches = batch_incremental(data,config['batch_size'])
        val_batches =  batch_iter_val(data, config['batch_size'])

        total_loss = 0
        train_total_corr=0
        train_total_pred=0

        for batch in train_batches:
            #current_step = tf.train.global_step(sess, global_step)

            (u1, v1, u2, v2, u3, v3, lu1, lv1, lu2, w_ll, w_lu, w_uu, c_ull, c_vll, c_ulu) = batch

            _, loss, tc, tp = sess.run([opt_op, loss_function, train_corr_sum,train_pred_num],
                                    feed_dict=
                                    {in_u1: u1,
                                     in_v1: v1,
                                     in_u2: u2,
                                     in_v2: v2,
                                     in_u3: u3,
                                     in_v3: v3,
                                     labels_u1: lu1,
                                     labels_v1: lv1,
                                     labels_u2: lu2,
                                     weights_ll: w_ll,
                                     weights_lu: w_lu,
                                     weights_uu: w_uu,
                                     cu1: c_ull,
                                     cv1: c_vll,
                                     cu2: c_ulu})

            train_total_corr+=tc
            train_total_pred+=tp

            total_loss += loss

        train_accuracy=train_total_corr/train_total_pred
        train_accuracies.append(train_accuracy)

        val_accuracy=calculate_accuracy(sess,test_inputs,test_labels,test_corr_sum,test_pred_num,val_batches)
        val_accuracies.append(val_accuracy)


        print("Epoch {0} Loss: {1} Train accuracy {2} Val Accuracy {3}".format(epoch, total_loss, train_accuracy,val_accuracy))

    test_batches = batch_iter_test(data, config['batch_size'])
    test_accuracy=calculate_accuracy(sess, test_inputs, test_labels, test_corr_sum, test_pred_num, test_batches)

    print("Test Accuracy {0}".format(test_accuracy))

    from GNN.Utils import plot_train_val_accuracy
    filename = config['dataset_name'] + '_NGM_Keras_' + 'epoch_' + str(epochs) + '_class_' + str(config['out_features'])
    plot_train_val_accuracy(config['output_path'],{'name': filename, 'train_accs': train_accuracies, 'val_accs': val_accuracies})

    save_file = config['output_path'] + 'NGM_Keras.h5'
    model.save(save_file)

    y_softmax = model.predict(data.Feature[data.test_index])
    y_test_1d = np.argmax(data.Label[data.test_index], axis=1)
    y_pred_1d = np.argmax(y_softmax, axis=1)

    from GNN.Utils import draw_confusion_matrix
    filename = config['output_path'] + "/NGM_Keras" + str(epochs) + "_CM.png"
    draw_confusion_matrix(y_test_1d, y_pred_1d, data.classname, filename)

    return

def load_saved(config,data):

    save_file = config['output_path'] + 'NGM_Keras.h5'
    model = tf.keras.models.load_model(save_file)

    y_softmax = model.predict(data.Feature[data.test_index])
    y_test_1d = np.argmax(data.Label[data.test_index], axis=1)
    y_pred_1d = np.argmax(y_softmax, axis=1)

    from GNN.Utils import draw_confusion_matrix
    filename = config['output_path'] + "/NGM_Keras_CM.png"
    draw_confusion_matrix(y_test_1d, y_pred_1d, data.classname, filename)

    return



def learn(config,data):
    start = timeit.default_timer()
    train(config,data)
    end = timeit.default_timer()
    print('Time : {0}'.format(end - start))


if __name__ == '__main__':
    start = timeit.default_timer()

    path="/Users/siddharthashankardas/Purdue/Dataset/Karate/"

    from GNN_configuration import getSettings, get_dataset
    load_config={
        "input_path":path,
        "labeled_only":True,
        "dataset_name":"karate"
    }
    data = get_dataset(load_config)

    print(data)

    gnn_settings = getSettings(load_config['dataset_name'])
    gnn_settings['output_path'] = path

    train(gnn_settings,data)
    #load_saved(gnn_settings,data)

    end = timeit.default_timer()
    print('Time : {0}'.format(end - start))
