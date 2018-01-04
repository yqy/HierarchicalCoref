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
import net as network
import evaluation
import utils

import cPickle
sys.setrecursionlimit(1000000)

torch.cuda.set_device(args.gpu)
 
def performance(doc,network_model,ana_network=None):

    test_document = []

    score_softmax = nn.Softmax()
    
    cluster_info = []
    new_cluster_num = 0 
    cluster_info.append(-1)
    gold_ana_result = []
    predict_ana_result = []
    aaa = 0

    this_pair_network = network_model
    if ana_network is None:
        this_ana_network = network_model
    else:
        this_ana_network = ana_network

    for data in doc.rl_case_generater(shuffle=False):
        
        this_doc = doc
        
        mention_word_index, mention_span, candi_word_index,candi_span,feature_pair,pair_antecedents,pair_anaphors,\
        target,positive,negative,anaphoricity_word_indexs, anaphoricity_spans, anaphoricity_features, anaphoricity_target,rl,candi_ids_return = data

        gold_ana_result += anaphoricity_target.tolist()

        mention_index = autograd.Variable(torch.from_numpy(mention_word_index).type(torch.cuda.LongTensor))
        mention_span = autograd.Variable(torch.from_numpy(mention_span).type(torch.cuda.FloatTensor))
        candi_index = autograd.Variable(torch.from_numpy(candi_word_index).type(torch.cuda.LongTensor))
        candi_spans = autograd.Variable(torch.from_numpy(candi_span).type(torch.cuda.FloatTensor))
        pair_feature = autograd.Variable(torch.from_numpy(feature_pair).type(torch.cuda.FloatTensor))
        anaphors = autograd.Variable(torch.from_numpy(pair_anaphors).type(torch.cuda.LongTensor))
        antecedents = autograd.Variable(torch.from_numpy(pair_antecedents).type(torch.cuda.LongTensor))

        anaphoricity_index = autograd.Variable(torch.from_numpy(anaphoricity_word_indexs).type(torch.cuda.LongTensor))
        anaphoricity_span = autograd.Variable(torch.from_numpy(anaphoricity_spans).type(torch.cuda.FloatTensor))
        anaphoricity_feature = autograd.Variable(torch.from_numpy(anaphoricity_features).type(torch.cuda.FloatTensor))

        output, pair_score = this_pair_network.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_span,candi_index,candi_spans,pair_feature,anaphors,antecedents)

        ana_output, ana_score = this_ana_network.forward_anaphoricity(nnargs["word_embedding_dimention"], anaphoricity_index, anaphoricity_span, anaphoricity_feature)
        ana_pair_output, ana_pair_score = this_ana_network.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_span,candi_index,candi_spans,pair_feature,anaphors,antecedents)

        reindex = autograd.Variable(torch.from_numpy(rl["reindex"]).type(torch.cuda.LongTensor))

        scores_reindex = torch.transpose(torch.cat((pair_score,ana_score),1),0,1)[reindex]
        ana_scores_reindex = torch.transpose(torch.cat((ana_pair_score,ana_score),1),0,1)[reindex]

        for s,e in zip(rl["starts"],rl["ends"]):
            #action_prob: scores_reindex[s:e][1]
            ana_score = score_softmax(torch.transpose(ana_scores_reindex[s:e],0,1)).data.cpu().numpy()[0]
            ac_list = ana_score.tolist()
            this_action = ac_list.index(max(ac_list)) 
            
            if this_action == len(ana_score)-1:
                aaa += 1
                should_cluster = new_cluster_num
                new_cluster_num += 1
                predict_ana_result.append(0)
            else:
                new_score = score_softmax(torch.transpose(scores_reindex[s:e-1],0,1)).data.cpu().numpy()[0]
                #ac_list = new_score.tolist()
                ac_list = (new_score*ana_score[:-1]).tolist()
                
                #new_action = ac_list.index(max(ac_list[-400:]))
                new_action = ac_list.index(max(ac_list))
                should_cluster = cluster_info[new_action]
                predict_ana_result.append(1)
            cluster_info.append(should_cluster)

        if rl["end"] == True:
            ev_document = utils.get_evaluation_document(cluster_info,this_doc.gold_chain[rl["did"]],candi_ids_return,new_cluster_num)
            test_document.append(ev_document)
            cluster_info = []
            new_cluster_num = 0 
            cluster_info.append(-1)

    ana_result = average_precision_score(numpy.array(gold_ana_result), numpy.array(predict_ana_result))
 
    metrics = evaluation.Output_Result(test_document)
    r,p,f = metrics["muc"]
    print "MUC: recall: %f precision: %f  f1: %f"%(r,p,f)
    r,p,f = metrics["b3"]
    print "B3: recall: %f precision: %f  f1: %f"%(r,p,f)
    r,p,f = metrics["ceaf"]
    print "CEAF: recall: %f precision: %f  f1: %f"%(r,p,f)
    metrics["ana"] = ana_result
    print "AVE",metrics["average"]

    return metrics

if __name__ == "__main__":
    DIR = args.DIR
    network_file = "./model/pretrain/network_model_rl_worker.7"
    #network_file = "./model/network_model_pretrain.top.best"
    network_model = torch.load(network_file)

    ana_network_file = "./model/pretrain/network_model_rl_manager.7"
    #ana_network_file = "./model/network_model_pretrain.top.best"
    ana_network_model = torch.load(ana_network_file)

    reduced=""
    if args.reduced == 1:
        reduced="_reduced"

    #dev_docs = DataReader.DataGnerater("dev"+reduced)
    test_docs = DataReader.DataGnerater("test"+reduced)

    #metric = performance(test_docs,network_model,ana_network_model)
    metric = performance(test_docs,network_model,ana_network_model)
    print "Ave",metric["average"]

    #network_file = "./model/network_model_pretrain.top.best"
    #network_model = torch.load(network_file)

    #ana_network_file = "./model/network_model_pretrain.top.best"
    #ana_network_model = torch.load(ana_network_file)

    #reduced=""
    #if args.reduced == 1:
    #    reduced="_reduced"

    #metric = performance(test_docs,network_model,ana_network_model)
    #print "Ave",metric["average"]
