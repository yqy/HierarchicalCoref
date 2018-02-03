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
from torch.optim import lr_scheduler

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

    best_network_file = "./model/network_model_pretrain.best.top.pair"
    print >> sys.stderr,"Read model from ",best_network_file
    best_network_model = torch.load(best_network_file)

    embedding_matrix = numpy.load(embedding_file)
    "Building torch model"
    network_model = network.Network(nnargs["pair_feature_dimention"],nnargs["mention_feature_dimention"],nnargs["word_embedding_dimention"],nnargs["span_dimention"],1000,nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix).cuda()
    net_copy(network_model,best_network_model)

    best_network_file = "./model/network_model_pretrain.best.top.ana"
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
    #lr = 0.000005
    lr = 0.000002
    #lr = 0.0000009
    dropout_rate = 0.5
    shuffle = True
    times = 0

    reinforce = True

    model_save_dir = "./model/reinforce/"
    utils.mkdir(model_save_dir)

    score_softmax = nn.Softmax()
    optimizer = optim.RMSprop(network_model.parameters(), lr=lr, eps = 1e-6)
    ana_optimizer = optim.RMSprop(ana_network.parameters(), lr=lr, eps = 1e-6)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    ana_scheduler = lr_scheduler.StepLR(ana_optimizer, step_size=15, gamma=0.5)
   
    for echo in range(30):

        start_time = timeit.default_timer()
        print "Pretrain Epoch:",echo

        scheduler.step()
        ana_scheduler.step()

        train_docs = utils.load_pickle(args.DOCUMENT + 'train_docs.pkl')

        docs_by_id = {doc.did: doc for doc in train_docs}
       
        print >> sys.stderr,"Link docs ..."
        tmp_data = []
        path = []
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

            output, pair_score = network_model.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_spans,candi_index,candi_spans,pair_feature,anaphors,antecedents,0.0)
            ana_output, ana_score = ana_network.forward_anaphoricity(nnargs["word_embedding_dimention"], anaphoricity_index, anaphoricity_span, anaphoricity_feature, 0.0)
            ana_pair_output, ana_pair_score = ana_network.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_spans,candi_index,candi_spans,pair_feature,anaphors,antecedents, 0.0)

            reindex = autograd.Variable(torch.from_numpy(rl["reindex"]).type(torch.cuda.LongTensor))

            scores_reindex = torch.transpose(torch.cat((pair_score,ana_score),1),0,1)[reindex]
            ana_scores_reindex = torch.transpose(torch.cat((ana_pair_score,ana_score),1),0,1)[reindex]

            doc = docs_by_id[rl['did']]

            for s,e in zip(rl["starts"],rl["ends"]):
                score = score_softmax(torch.transpose(ana_scores_reindex[s:e],0,1)).data.cpu().numpy()[0]
                pair_score = score_softmax(torch.transpose(scores_reindex[s:e-1],0,1)).data.cpu().numpy()[0]

                ana_action = utils.sample_action(score)
                if ana_action == (e-s-1):
                    action = ana_action
                else:
                    pair_action = utils.sample_action(pair_score*score[:-1])
                    action = pair_action
                path.append(action)
                link = action
                m1, m2 = rl['ids'][s + link]
                doc.link(m1, m2)

            tmp_data.append((mention_word_index, mention_span, candi_word_index,candi_span,feature_pair,pair_antecedents,pair_anaphors,target,positive,negative,anaphoricity_word_indexs, anaphoricity_spans, anaphoricity_features, anaphoricity_target,rl,candi_ids_return))
                
            if rl["end"] == True:
                doc = docs_by_id[rl['did']]
                reward = doc.get_f1()
                inside_index = 0
                for mention_word_index, mention_span, candi_word_index,candi_span,feature_pair,pair_antecedents,pair_anaphors,target,positive,negative,anaphoricity_word_indexs, anaphoricity_spans, anaphoricity_features, anaphoricity_target,rl,candi_ids_return in tmp_data:

                    for (start, end) in zip(rl['starts'], rl['ends']):
                        ids = rl['ids'][start:end]
                        ana = ids[0, 1]
                        old_ant = doc.ana_to_ant[ana]
                        doc.unlink(ana)
                        costs = rl['costs'][start:end]
                        for ant_ind in range(end - start):
                            costs[ant_ind] = doc.link(ids[ant_ind, 0], ana, hypothetical=True, beta=1)
                        doc.link(old_ant, ana) 

                    cost = 0.0
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
                    i = inside_index
                    for s,e in zip(rl["starts"],rl["ends"]):
                        costs = rl["costs"][s:e]
                        costs = autograd.Variable(torch.from_numpy(costs).type(torch.cuda.FloatTensor))
                        score = torch.squeeze(score_softmax(torch.transpose(ana_scores_reindex[s:e],0,1)))
                        baseline = torch.sum(score*costs) 

                        action = path[i]
                        this_cost = torch.log(score[action])*-1.0*(reward-baseline)
                        
                        if ana_loss is None:
                            ana_loss = this_cost
                        else:
                            ana_loss += this_cost
                        i += 1
                    ana_loss.backward()
                    torch.nn.utils.clip_grad_norm(ana_network.parameters(), 5.0)
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
                    i = inside_index
                    index = 0
                    for s,e in zip(rl["starts"],rl["ends"]):
                        action = path[i]
                        if (not (action == (e-s-1))) and (anaphoricity_target[index] == 1):
                            costs = rl["costs"][s:e-1]
                            costs = autograd.Variable(torch.from_numpy(costs).type(torch.cuda.FloatTensor))
                            score = torch.squeeze(score_softmax(torch.transpose(scores_reindex[s:e-1],0,1)))
                            baseline = torch.sum(score*costs)
                            this_cost = torch.log(score[action])*-1.0*(reward-baseline)
                            if pair_loss is None:
                                pair_loss = this_cost
                            else:
                                pair_loss += this_cost
                        i += 1
                        index += 1
                    if pair_loss is not None:
                        pair_loss.backward()
                        torch.nn.utils.clip_grad_norm(network_model.parameters(), 5.0)
                        optimizer.step()
                    inside_index = i

                tmp_data = []
                path = []
                        
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
