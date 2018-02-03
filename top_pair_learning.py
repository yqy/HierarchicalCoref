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
import performance

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

    best_network_file = "./model/network_model_pretrain.best.pair"
    print >> sys.stderr,"Read model from",best_network_file
    best_network_model = torch.load(best_network_file)
        
    embedding_matrix = numpy.load(embedding_file)

    "Building torch model"
    network_model = network.Network(nnargs["pair_feature_dimention"],nnargs["mention_feature_dimention"],nnargs["word_embedding_dimention"],nnargs["span_dimention"],1000,nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix).cuda()
    print >> sys.stderr,"save model ..."

    net_copy(network_model,best_network_model)

    reduced=""
    if args.reduced == 1:
        reduced="_reduced"

    print >> sys.stderr,"prepare data for train ..."
    train_docs = DataReader.DataGnerater("train"+reduced)
    print >> sys.stderr,"prepare data for dev and test ..."
    dev_docs = DataReader.DataGnerater("dev"+reduced)
    test_docs = DataReader.DataGnerater("test"+reduced)


    l2_lambda = 1e-6
    lr = 0.0001
    dropout_rate = 0.5
    shuffle = True
    times = 0
    best_thres = 0.5

    model_save_dir = "./model/"
   
    last_cost = 0.0
    all_best_results = {
        'thresh': 0.0,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
        }
    
    optimizer = optim.RMSprop(network_model.parameters(), lr=lr, eps=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)
  
    for echo in range(100):

        start_time = timeit.default_timer()
        print "Pretrain Epoch:",echo
    
        scheduler.step()

        pair_cost_this_turn = 0.0
        ana_cost_this_turn = 0.0

        pair_nums = 0
        ana_nums = 0

        pos_num = 0
        neg_num = 0
        inside_time = 0.0
        

        for data in train_docs.train_generater(shuffle=shuffle,top=True):
            
            mention_word_index, mention_span, candi_word_index,candi_span,feature_pair,pair_antecedents,pair_anaphors,\
            target,positive,negative,anaphoricity_word_indexs, anaphoricity_spans, anaphoricity_features, anaphoricity_target,top_x = data
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

            reindex = autograd.Variable(torch.from_numpy(top_x["score_index"]).type(torch.cuda.LongTensor))


            start_index = autograd.Variable(torch.from_numpy(top_x["starts"]).type(torch.cuda.LongTensor))
            end_index = autograd.Variable(torch.from_numpy(top_x["ends"]).type(torch.cuda.LongTensor))

            top_gold = autograd.Variable(torch.from_numpy(top_x["top_gold"]).type(torch.cuda.FloatTensor))

            anaphoricity_gold = anaphoricity_target.tolist()
            ana_lable = autograd.Variable(torch.cuda.FloatTensor([anaphoricity_gold]))

            optimizer.zero_grad()

            output,output_reindex = network_model.forward_top_pair(nnargs["word_embedding_dimention"],mention_index,mention_span,candi_index,candi_spans,pair_feature,anaphors,antecedents,reindex,start_index,end_index,dropout_rate)
            loss = F.binary_cross_entropy(output,top_gold,size_average=False)/train_docs.scale_factor_top

            loss_all = loss   
            
            loss_all.backward()
            pair_cost_this_turn += loss.data[0]
            optimizer.step()

        end_time = timeit.default_timer()
        print >> sys.stderr, "PreTrain",echo,"Pair total cost:",pair_cost_this_turn
        print >> sys.stderr, "PreTRAINING Use %.3f seconds"%(end_time-start_time)
        print >> sys.stderr, "Learning Rate",lr

        gold = []
        predict = []

        ana_gold = []
        ana_predict = []

        for data in dev_docs.train_generater(shuffle=False,top = True):
            
            mention_word_index, mention_span, candi_word_index,candi_span,feature_pair,pair_antecedents,pair_anaphors,\
            target,positive,negative, anaphoricity_word_indexs, anaphoricity_spans, anaphoricity_features, anaphoricity_target, top_x = data
         
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

            reindex = autograd.Variable(torch.from_numpy(top_x["score_index"]).type(torch.cuda.LongTensor))
            start_index = autograd.Variable(torch.from_numpy(top_x["starts"]).type(torch.cuda.LongTensor))
            end_index = autograd.Variable(torch.from_numpy(top_x["ends"]).type(torch.cuda.LongTensor))

            gold += top_x["top_gold"].tolist()
            ana_gold += anaphoricity_target.tolist()
        
            output,output_reindex = network_model.forward_top_pair(nnargs["word_embedding_dimention"],mention_index,mention_span,candi_index,candi_spans,pair_feature,anaphors,antecedents,reindex,start_index,end_index,0.0)

            predict += output.data.cpu().numpy().tolist()

        
        gold = numpy.array(gold,dtype=numpy.int32)
        predict = numpy.array(predict)

        best_results = {
            'thresh': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

        thresh_list = [0.3,0.35,0.4,0.45,0.5,0.55,0.6]
        for thresh in thresh_list:
            evaluation_results = get_metrics(gold, predict, thresh)
            if evaluation_results["f1"] >= best_results["f1"]:
                best_results = evaluation_results
 
        print "Pair accuracy: %f and Fscore: %f with thresh: %f"\
                %(best_results["accuracy"],best_results["f1"],best_results["thresh"])
        sys.stdout.flush() 

        if best_results["f1"] >= all_best_results["f1"]:
            all_best_results = best_results
            print >> sys.stderr, "New High Result, Save Model"
            torch.save(network_model, model_save_dir+"network_model_pretrain.best.top.pair")

def get_metrics(gold, predict, thresh):
    pred = np.clip(np.floor(predict / thresh), 0, 1)
    p, r = (0, 0) if pred.sum() == 0 else \
    (precision_score(gold, pred), recall_score(gold, pred))
    return {
        'thresh': thresh,
        'accuracy': average_precision_score(gold, predict),
        'precision': p,
        'recall': r,
        'f1': 0 if p == 0 or r == 0 else 2 * p * r / (p + r)
    } 

if __name__ == "__main__":
    main()
