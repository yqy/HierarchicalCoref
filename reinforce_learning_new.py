#coding=utf8

import sys
import os
import json
import random
import numpy
import timeit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
import torch.optim as optim

from sklearn.metrics import accuracy_score, average_precision_score, precision_score,recall_score

from conf import *

import DataReader
import evaluation
import net as network
import utils
import performance

from document import *

import cPickle
sys.setrecursionlimit(1000000)

print >> sys.stderr, "PID", os.getpid()

torch.cuda.set_device(args.gpu)

def net_copy(net,copy_from_net):
    mcp = list(net.parameters())
    mp = list(copy_from_net.parameters())
    n = len(mcp)
    for i in range(0, n): 
        mcp[i].data[:] = mp[i].data[:]
 
def main():

    DIR = args.DIR
    embedding_file = args.embedding_dir

    best_network_file = "./model/network_model_pretrain.best.top"
    print >> sys.stderr,"Read model from ",best_network_file
    best_network_model = torch.load(best_network_file)

    embedding_matrix = numpy.load(embedding_file)
    "Building torch model"
    network_model = network.Network(nnargs["pair_feature_dimention"],nnargs["mention_feature_dimention"],nnargs["word_embedding_dimention"],nnargs["span_dimention"],1000,nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix).cuda()
    net_copy(network_model,best_network_model)

    best_network_file = "./model/network_model_pretrain.best.top"
    print >> sys.stderr,"Read model from ",best_network_file
    best_network_model = torch.load(best_network_file)

    ana_network = network.Network(nnargs["pair_feature_dimention"],nnargs["mention_feature_dimention"],nnargs["word_embedding_dimention"],nnargs["span_dimention"],1000,nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix).cuda()
    net_copy(ana_network,best_network_model)

    reduced=""
    if args.reduced == 1:
        reduced="_reduced"

    print >> sys.stderr,"prepare data for train ..."
    train_docs_iter = DataReader.DataGnerater("train"+reduced)
    print >> sys.stderr,"prepare data for dev and test ..."
    dev_docs_iter = DataReader.DataGnerater("dev"+reduced)
    test_docs_iter = DataReader.DataGnerater("test"+reduced)

    print "Performance after pretraining..."
    print "DEV"
    metric = performance.performance(dev_docs_iter,network_model,ana_network) 
    print "Average:",metric["average"]
    print "TEST"
    metric = performance.performance(test_docs_iter,network_model,ana_network) 
    print "Average:",metric["average"]
    print "***"
    print
    sys.stdout.flush()

    l2_lambda = 1e-6
    #lr = 0.00001
    lr = 0.000005
    #lr = 0.0000009
    dropout_rate = 0.5
    shuffle = True
    times = 0

    reinforce = True

    model_save_dir = "./model/reinforce/"
    utils.mkdir(model_save_dir)

    score_softmax = nn.Softmax()
   
    for echo in range(30):

        start_time = timeit.default_timer()
        print "Pretrain Epoch:",echo

        train_docs = utils.load_pickle(args.DOCUMENT + 'train_docs.pkl')

        docs_by_id = {doc.did: doc for doc in train_docs}
       
        print >> sys.stderr,"Link docs ..."
        tmp_data = []
        for data in train_docs_iter.rl_case_generater(shuffle=True):
            mention_word_index, mention_span, candi_word_index,candi_span,feature_pair,pair_antecedents,pair_anaphors,\
            target,positive,negative,anaphoricity_word_indexs, anaphoricity_spans, anaphoricity_features, anaphoricity_target,rl,candi_ids_return = data

            mention_index = autograd.Variable(torch.from_numpy(mention_word_index).type(torch.cuda.LongTensor))
            mention_spans = autograd.Variable(torch.from_numpy(mention_span).type(torch.cuda.FloatTensor))
            candi_index = autograd.Variable(torch.from_numpy(candi_word_index).type(torch.cuda.LongTensor))
            candi_spans = autograd.Variable(torch.from_numpy(candi_span).type(torch.cuda.FloatTensor))
            pair_feature = autograd.Variable(torch.from_numpy(feature_pair).type(torch.cuda.FloatTensor))
            anaphors = autograd.Variable(torch.from_numpy(pair_anaphors).type(torch.cuda.LongTensor))
            antecedents = autograd.Variable(torch.from_numpy(pair_antecedents).type(torch.cuda.LongTensor))

            anaphoricity_index = autograd.Variable(torch.from_numpy(anaphoricity_word_indexs).type(torch.cuda.LongTensor))
            anaphoricity_span = autograd.Variable(torch.from_numpy(anaphoricity_spans).type(torch.cuda.FloatTensor))
            anaphoricity_feature = autograd.Variable(torch.from_numpy(anaphoricity_features).type(torch.cuda.FloatTensor))

            output, pair_score = network_model.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_spans,candi_index,candi_spans,pair_feature,anaphors,antecedents,dropout_rate)
            ana_output, ana_score = ana_network.forward_anaphoricity(nnargs["word_embedding_dimention"], anaphoricity_index, anaphoricity_span, anaphoricity_feature, dropout_rate)
            ana_pair_output, ana_pair_score = ana_network.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_spans,candi_index,candi_spans,pair_feature,anaphors,antecedents,dropout_rate)

            reindex = autograd.Variable(torch.from_numpy(rl["reindex"]).type(torch.cuda.LongTensor))

            scores_reindex = torch.transpose(torch.cat((pair_score,ana_score),1),0,1)[reindex]
            ana_scores_reindex = torch.transpose(torch.cat((ana_pair_score,ana_score),1),0,1)[reindex]

            scores4cost = []

            for s,e in zip(rl["starts"],rl["ends"]):
                score = score_softmax(torch.transpose(ana_scores_reindex[s:e],0,1)).data.cpu().numpy()[0]
                this_action = utils.choose_action(score)
                pair_score = score_softmax(torch.transpose(scores_reindex[s:e-1],0,1)).data.cpu().numpy()[0]
                if this_action == len(score)-1 :
                    ana_score = 1.0
                else:
                    ana_score = 0.0 
                scores4cost += (pair_score*score[:-1]).tolist()
                scores4cost += [ana_score]

            scores4cost = numpy.array(scores4cost)

            update_doc(docs_by_id[rl['did']], rl, scores4cost)

            tmp_data.append((mention_word_index, mention_span, candi_word_index,candi_span,feature_pair,pair_antecedents,pair_anaphors,target,positive,negative,anaphoricity_word_indexs, anaphoricity_spans, anaphoricity_features, anaphoricity_target,rl,candi_ids_return))
                
            if rl["end"] == True:
                docs_by_id_this = {doc.did: doc for doc in train_docs}
                for mention_word_index, mention_span, candi_word_index,candi_span,feature_pair,pair_antecedents,pair_anaphors,target,positive,negative,anaphoricity_word_indexs, anaphoricity_spans, anaphoricity_features, anaphoricity_target,rl,candi_ids_return in tmp_data:

                    doc = docs_by_id_this[rl['did']]
                    doc_weight = (len(doc.mention_to_gold) + len(doc.mentions)) / 10.0
                    for (start, end) in zip(rl['starts'], rl['ends']):
                        ids = rl['ids'][start:end]
                        ana = ids[0, 1]
                        old_ant = doc.ana_to_ant[ana]
                        doc.unlink(ana)
                        costs = rl['costs'][start:end]
                        for ant_ind in range(end - start):
                            costs[ant_ind] = doc.link(ids[ant_ind, 0], ana, hypothetical=True, beta=1)
                        doc.link(old_ant, ana) 

                        costs -= costs.max()
                        costs *= -doc_weight

                    cost = 0.0
                    optimizer = optim.RMSprop(network_model.parameters(), lr=lr, eps = 1e-5)
                    ana_optimizer = optim.RMSprop(ana_network.parameters(), lr=lr, eps = 1e-5)

                    mention_index = autograd.Variable(torch.from_numpy(mention_word_index).type(torch.cuda.LongTensor))
                    mention_spans = autograd.Variable(torch.from_numpy(mention_span).type(torch.cuda.FloatTensor))
                    candi_index = autograd.Variable(torch.from_numpy(candi_word_index).type(torch.cuda.LongTensor))
                    candi_spans = autograd.Variable(torch.from_numpy(candi_span).type(torch.cuda.FloatTensor))
                    pair_feature = autograd.Variable(torch.from_numpy(feature_pair).type(torch.cuda.FloatTensor))
                    anaphors = autograd.Variable(torch.from_numpy(pair_anaphors).type(torch.cuda.LongTensor))
                    antecedents = autograd.Variable(torch.from_numpy(pair_antecedents).type(torch.cuda.LongTensor))
                    anaphoricity_index = autograd.Variable(torch.from_numpy(anaphoricity_word_indexs).type(torch.cuda.LongTensor))
                    anaphoricity_span = autograd.Variable(torch.from_numpy(anaphoricity_spans).type(torch.cuda.FloatTensor))
                    anaphoricity_feature = autograd.Variable(torch.from_numpy(anaphoricity_features).type(torch.cuda.FloatTensor))
        
                    ana_output, ana_score = ana_network.forward_anaphoricity(nnargs["word_embedding_dimention"], anaphoricity_index, anaphoricity_span, anaphoricity_feature, dropout_rate)
                    ana_pair_output, ana_pair_score = ana_network.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_spans,candi_index,candi_spans,pair_feature,anaphors,antecedents,dropout_rate)
        
                    reindex = autograd.Variable(torch.from_numpy(rl["reindex"]).type(torch.cuda.LongTensor))
        
                    ana_scores_reindex = torch.transpose(torch.cat((ana_pair_score,ana_score),1),0,1)[reindex]
        
                    ana_optimizer.zero_grad()
                    ana_loss = None
                    for s,e in zip(rl["starts"],rl["ends"]):
                        reward_list = rl["costs"][s:e]
                        #reward_list = reward_list - np.mean(reward_list)
                        costs = autograd.Variable(torch.from_numpy(reward_list).type(torch.cuda.FloatTensor))
                        ana_scores_softmax = score_softmax(torch.transpose(ana_scores_reindex[s:e],0,1))
                        this_cost = torch.sum(ana_scores_softmax*costs)
                        
                        if ana_loss is None:
                            ana_loss = this_cost
                        else:
                            ana_loss += this_cost
                    ana_loss.backward()
                    ana_optimizer.step()
        
                    mention_index = autograd.Variable(torch.from_numpy(mention_word_index).type(torch.cuda.LongTensor))
                    mention_spans = autograd.Variable(torch.from_numpy(mention_span).type(torch.cuda.FloatTensor))
                    candi_index = autograd.Variable(torch.from_numpy(candi_word_index).type(torch.cuda.LongTensor))
                    candi_spans = autograd.Variable(torch.from_numpy(candi_span).type(torch.cuda.FloatTensor))
                    pair_feature = autograd.Variable(torch.from_numpy(feature_pair).type(torch.cuda.FloatTensor))
                    anaphors = autograd.Variable(torch.from_numpy(pair_anaphors).type(torch.cuda.LongTensor))
                    antecedents = autograd.Variable(torch.from_numpy(pair_antecedents).type(torch.cuda.LongTensor))
        
                    anaphoricity_index = autograd.Variable(torch.from_numpy(anaphoricity_word_indexs).type(torch.cuda.LongTensor))
                    anaphoricity_span = autograd.Variable(torch.from_numpy(anaphoricity_spans).type(torch.cuda.FloatTensor))
                    anaphoricity_feature = autograd.Variable(torch.from_numpy(anaphoricity_features).type(torch.cuda.FloatTensor))
        
                    output, pair_score = network_model.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_spans,candi_index,candi_spans,pair_feature,anaphors,antecedents,dropout_rate)
        
                    ana_output, ana_score = ana_network.forward_anaphoricity(nnargs["word_embedding_dimention"], anaphoricity_index, anaphoricity_span, anaphoricity_feature, dropout_rate)
        
                    reindex = autograd.Variable(torch.from_numpy(rl["reindex"]).type(torch.cuda.LongTensor))
        
                    scores_reindex = torch.transpose(torch.cat((pair_score,ana_score),1),0,1)[reindex]
        
                    pair_loss = None
                    optimizer.zero_grad()
                    index = 0
                    for s,e in zip(rl["starts"],rl["ends"]):
                        if anaphoricity_target[index] == 1:
                            reward_list = rl["costs"][s:e-1]
                            #reward_list = reward_list - np.mean(reward_list)
                            costs = autograd.Variable(torch.from_numpy(reward_list).type(torch.cuda.FloatTensor))
        
                            pair_scores_softmax = score_softmax(torch.transpose(scores_reindex[s:e-1],0,1))
                            this_cost = torch.mean(pair_scores_softmax*costs)
                        
                            if pair_loss is None:
                                pair_loss = this_cost
                            else:
                                pair_loss += this_cost
                        index += 1
                    if pair_loss is not None:
                        pair_loss.backward()
                        optimizer.step()
                tmp_data = []
                        
        end_time = timeit.default_timer()
        print >> sys.stderr, "TRAINING Use %.3f seconds"%(end_time-start_time)
        print >> sys.stderr, "cost:",cost
        print >> sys.stderr,"save model ..."
        torch.save(network_model, model_save_dir+"network_model_rl_worker.%d"%echo)
        torch.save(ana_network, model_save_dir+"network_model_rl_manager.%d"%echo)
        
        print "DEV"
        metric = performance.performance(dev_docs_iter,network_model,ana_network) 
        print "Average:",metric["average"]
        print "DEV Ana: ",metric["ana"]
        print "TEST"
        metric = performance.performance(test_docs_iter,network_model,ana_network) 
        print "Average:",metric["average"]
        print "TEST Ana: ",metric["ana"]
        print

        sys.stdout.flush()

if __name__ == "__main__":
    main()
