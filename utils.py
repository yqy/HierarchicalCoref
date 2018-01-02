#coding=utf8

import sys
import os
import json
import random
import numpy
import timeit

from conf import *

import evaluation

import cPickle
sys.setrecursionlimit(1000000)

random.seed(args.random_seed)

def sample_action(action_probability):
    
    ac = action_probability
    ac = ac/ac.sum()
    action = numpy.random.choice(numpy.arange(len(ac)),p=ac)
    return action

def choose_action(action_probability):
    #print action_probability
    ac_list = list(action_probability)
    action = ac_list.index(max(ac_list))
    return action

def get_reward(cluster_info,gold_info,max_cluster_num):
    ev_document = get_evaluation_document(cluster_info,gold_info,max_cluster_num)
    p,r,f = evaluation.evaluate_documents([ev_document],evaluation.b_cubed)
    return p,r,f

def get_reward_average(cluster_info,gold_info,max_cluster_num,index,max_cluster_index,candi_ids):
    # build new cluster
    new_cluster_prefix = cluster_info[:index]
    new_cluster_postfix = cluster_info[index+1:]

    el = []
    
    for cluster_num in range(max_cluster_index):
        new_cluster_info = new_cluster_prefix + [cluster_num] + new_cluster_postfix 
        ev_document = get_evaluation_document(new_cluster_info,gold_info,candi_ids,max_cluster_num)
        el.append(ev_document)
    p,r,f = evaluation.evaluate_documents(el,evaluation.b_cubed)
    #p,r,f = evaluation.evaluate_documents([ev_document],evaluation.muc)
    #print >> sys.stderr, p,r,f
    return f


def get_evaluation_document(cluster_info,gold_info,doc_ids,max_cluster_num):
    predict = []
    predict_dict = {}
   
    for mention_num in range(len(cluster_info)):
        cluster_num = cluster_info[mention_num]
        predict_dict.setdefault(cluster_num,[])
        a = doc_ids[mention_num]
        predict_dict[cluster_num].append(a)
        #predict[cluster_num].append(mention_num)
    for k in sorted(predict_dict.keys()):
        predict.append(predict_dict[k])
    ev_document = evaluation.EvaluationDocument(gold_info,predict)
    return ev_document

def get_prf(predict,gold):
    if predict is not None:
        pr = 0
        should = 0
        r_in_r = 0
        for i in range(len(predict)):
            if not predict[i] == -1: ## a new cluster
                pr += 1 
                if gold[i] == 1:
                    r_in_r += 1
            if gold[i] == 1:
                should += 1
        if should == 0:
            return 0.0,0.0,0.0
        elif pr == 0:
            return 0.0,0.0,0.0
        else:
            p = float(r_in_r)/float(pr)
            r = float(r_in_r)/float(should)
            if (not p == 0) and (not r == 0):
                f = 2.0/(1.0/r+1.0/p)
            else:
                f = 0.0
            return p,r,f

# basic utils

def load_pickle(fname):
    with open(fname) as f:
        return cPickle.load(f)

def write_pickle(o, fname):
    with open(fname, 'w') as f:
        cPickle.dump(o, f, -1) 

def load_json_lines(fname):
    with open(fname) as f:
        for line in f:
            yield json.loads(line)

def lines_in_file(fname):
    return int(subprocess.check_output(
        ['wc', '-l', fname]).strip().split()[0])

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rmkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
