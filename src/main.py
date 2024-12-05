import sys
import numpy as np
import torch
from tqdm import tqdm
import random
sys.path.append("..")
import utils
from torch import nn
import json
from config import args
from model import PET
from datetime import datetime
from torch.utils import data as torch_data
from torch import distributed as dist
import os
from collections import defaultdict
from data_encoder import DataEncoder
from torch.utils.data import DataLoader
from datetime import datetime, timedelta, date
import dgl
from logging import getLogger
import pickle

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
accumulation_steps = 1

def train_and_validate(args, model, train_list, valid_list, test_list, graphs, num_nodes, num_users, model_state_file):
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    if utils.get_rank() == 0:
        print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\nstart training\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    optimizer = torch.optim.Adam(parallel_model.parameters(), lr=args.lr, weight_decay=5e-6)

    # the best recall@20
    best_recall = 0

    # read snapshot
    dict_snapshot = {}
    cache_file_name = '../../data/graphs/snapshot.json'

    with open(cache_file_name, 'r') as f:
        dict_snapshot = eval(f.read())

    for epoch in range(args.n_epoch):
        if utils.get_rank() == 0:
            print("\nepoch:"+str(epoch)+ ' Time: ' + datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S'))
        parallel_model.train()
        losses = list()
        train_dataloader = DataLoader(dataset=train_list, batch_size=args.batch_size)
        with tqdm(total=len(train_dataloader)) as t:
            for step, batch in enumerate(train_dataloader):
                #0.batch[0]: history_poi; 1.batch[1]: history_day_period; 2.batch[2]: history_poi_utc; 3.batch[3]: history_time_encoding
                #4.batch[4]: current_poi; 5.batch[5]: user; 6.batch[6]: target_poi; 7.batch[7]: target_day_period
                #8.batch[8]: current_poi_utc; batch[9]: target_poi_utc

                history_poi = torch.LongTensor(batch[0]).to(device)
                history_time_encoding = torch.LongTensor(batch[3]).to(device)
                history_g_list = []
                future_triple = []
                history_g_index = batch[1]
                last_index = -1

                # statisticize the temporal period of each trajectory flow graph
                g_time = []
                for step2, index in enumerate(history_g_index):
                    if index[0] == last_index:
                        last_index = index[0]
                        continue
                    else:
                        last_index = index[0]
                    if index[0].item() not in dict_snapshot.keys():
                        continue
                    day = np.array(dict_snapshot[index[0].item()])
                    g_start_head = day[:,3] >= batch[2][0][0]
                    g_start_tail = day[:,4] >= batch[2][0][0]
                    g_start = g_start_head * g_start_tail
                    g_end = day[:,4] < batch[9][0]
                    g_index = g_start * g_end
                    if(np.sum(g_index)) == 0:
                        continue
                    # construct temporal trajectory flow graphs
                    snapshot = np.asarray(day[g_index])
                    src, rel, dst, src_utc, dst_utc = snapshot.transpose()
                    src1 = src.astype(int)
                    rel1 = rel.astype(int)
                    dst1 = dst.astype(int)
                    g = dgl.graph(([],[]))
                    g.add_nodes(num_nodes)
                    g.add_edges(src1, dst1)
                    g.edata['type'] = torch.LongTensor(rel1)
                    history_g_list.append(g.to(device))
                    g_time.append(batch[3][step2])

                future_triple.append(batch[4].item())
                future_triple.append(batch[5].item())
                future_triple.append(batch[6].item())
                future_triple = torch.from_numpy(np.array(future_triple)).long().to(device)
                future_triple= future_triple.unsqueeze(0)
                history_poi = history_poi.unsqueeze(0)

                batch_future_all = utils.negative_sampling(future_triple, future_triple, args.negative_num, num_nodes, num_users, strict=True)
                pred= parallel_model(history_g_list, batch_future_all, history_poi, history_time_encoding, g_time)
                loss = model.get_loss(args, pred)
                losses.append(loss.item())
                t.set_description(desc="Epoch %i"%epoch)
                t.set_postfix(steps=step, loss=loss.data.item())
                t.update(1)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) 
                optimizer.step()
                optimizer.zero_grad()
                utils.synchronize()

            utils.synchronize()

            if utils.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                print("average binary cross entropy: {}".format(avg_loss))
                
            # evaluation
            if utils.get_rank() == 0:
                print("valid dataset eval:")
            recall_valid, f1_valid, ndcg_valid, mrr_valid = test(model, valid_list, num_users, num_nodes)

            if recall_valid >= best_recall:
                best_recall = recall_valid
                if utils.get_rank() == 0:
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'args': args}, model_state_file)
                    print("best_recall updated(epoch %d)!" %epoch)
                utils.synchronize()
                
                
            if utils.get_rank() == 0:
                print("\n---------------------------------")
            utils.synchronize()
    
    # testing
    if rank == 0 :
        print("\nFinal eval test dataset with best model:...")

    recall_test, f1_test, ndcg_test, mrr_test = test(model, test_list, num_users, num_nodes, mode="test", model_name=model_state_file)

    print(recall_test, f1_test, ndcg_test, mrr_test)

@torch.no_grad()
def test(model, test_list, num_users, num_nodes, mode="train", model_name = None):
    # Used to record evaluation results.
    intermediate_result = defaultdict(float)
    result = {}
    topk = [5, 10, 20]

    if mode == "test":
        # test mode: load parameter form file
        checkpoint = torch.load(model_name, map_location=device)
        checkpoint = torch.load(model_name)
        if utils.get_rank() == 0:
            print("\nLoad Model name: {}. Using best epoch : {}. \n\nargs:{}.".format(model_name, checkpoint['epoch'], checkpoint['args']))  # use best stat checkpoint
            print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\nstart test\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)

    model.eval()

    # read snapshot
    dict_snapshot = {}
    cache_file_name = '../../process/graphs/snapshot.json'
    with open(cache_file_name, 'r') as f:
        dict_snapshot = eval(f.read())

    future_loader = DataLoader(dataset=test_list, batch_size=args.batch_size)
    with tqdm(total=len(future_loader)) as t_test:
        for step, batch in enumerate(future_loader):
            history_poi = torch.LongTensor(batch[0]).to(device)
            history_time_encoding = torch.LongTensor(batch[3]).to(device)
            history_g_list = []
            future_triple = []
            history_g_index = batch[1]
            target_day_period = batch[7][0]
            last_index = -1

            # statisticize the temporal period of each trajectory flow graph
            g_time = []
            for step2, index in enumerate(history_g_index):
                if index[0] == last_index:
                    last_index = index[0]
                    continue
                else:
                    last_index = index[0]
                if dict_snapshot[index[0].item()] == None:
                    continue
                day = np.array(dict_snapshot[index[0].item()])
                g_start_head = day[:,3] >= batch[2][0][0]
                g_start_tail = day[:,4] >= batch[2][0][0]
                g_start = g_start_head * g_start_tail
                g_end = day[:,4] < batch[9][0]
                g_index = g_start * g_end
                if(np.sum(g_index)) == 0:
                    continue
                # construct temporal trajectory flow graphs
                snapshot = np.asarray(day[g_index])
                src, rel, dst, src_utc, dst_utc = snapshot.transpose()
                src1 = src.astype(int)
                rel1 = rel.astype(int)
                dst1 = dst.astype(int)
                g = dgl.graph(([],[]))
                g.add_nodes(num_nodes)
                g.add_edges(src1, dst1)
                g.edata['type'] = torch.LongTensor(rel1)
                history_g_list.append(g.to(device))
                g_time.append(batch[3][step2])

            future_triple.append(batch[4].item())
            future_triple.append(batch[5].item())
            future_triple.append(batch[6].item())
            future_triple = torch.from_numpy(np.array(future_triple)).long().to(device)
            future_triple= future_triple.unsqueeze(0)
            history_poi = history_poi.unsqueeze(0)

            # predict
            t_batch = utils.all_negative(num_nodes, future_triple)
            t_pred = model(history_g_list, t_batch, history_poi, history_time_encoding, g_time)
            # pos_t_index is the target POI
            pos_h_index, pos_r_index, pos_t_index = future_triple.t() 
        
            #Evaluation score
            total = len(future_triple)
            intermediate_result['total'] += total
            for idx in range(len(topk)):
                hit, ranking, dcg = utils.top_k(t_pred, pos_t_index, topk[idx])
                intermediate_result['hit' + str(topk[idx])] += hit
                intermediate_result['rank' + str(topk[idx])] += ranking
                intermediate_result['dcg' + str(topk[idx])] += dcg
            utils.synchronize()

            t_test.set_postfix(steps=step)
            t_test.update(1)

    utils.synchronize()

    # This is the end of prediction at 'future_sample_id' time
    # This is the end of prediction at test_set
    for k in topk:
        # Precision
        precision_key = 'Precision@{}'.format(k)
        precision = intermediate_result['hit' + str(k)] / (intermediate_result['total'] * k)
        result[precision_key] = precision
        # Recall
        recall_key = 'Recall@{}'.format(k)
        recall = intermediate_result['hit' + str(k)] / intermediate_result['total']
        result[recall_key] = recall
        # F1
        f1_key = 'F1@{}'.format(k)
        if precision + recall == 0:
            result[f1_key] = 0.0
        else:
            result[f1_key] = (2 * precision * recall) / (precision + recall)
        # MRR
        mrr_key = 'MRR@{}'.format(k)
        result[mrr_key] = intermediate_result['rank' + str(k)] / intermediate_result['total']
        # NDCG
        ndcg_key = 'NDCG@{}'.format(k)
        result[ndcg_key] = intermediate_result['dcg' + str(k)] / intermediate_result['total']

    print(json.dumps(result, indent=4))


    return result[recall_key], result[f1_key], result[ndcg_key], result[mrr_key]


if __name__ == '__main__':
    utils.set_rand_seed(2023)
    working_dir = utils.create_working_directory(args)

    # model name
    model_name = "bsize:{}-neg:{}-msg:{}-dim:{}+{}|{}|{}"\
        .format(args.batch_size, args.negative_num, args.message_func, args.input_dim, 
        args.hidden_dims, args.layer_norm, args.num_heads)

    # saved model name
    model_state_file = model_name

    # Initialize the training set, validation set, and test set.
    dataset = DataEncoder()
    graphs, train_data, eval_data, test_data = dataset.get_data(args.dataset)
    num_nodes = dataset.return_num_nodes(args.dataset)
    num_users = dataset.return_num_users(args.dataset)


    # model create
    model = PET(
        args.input_dim, 
        args.hidden_dims,
        num_nodes,
        num_users,
        message_func=args.message_func,
        short_cut=args.short_cut, 
        layer_norm=args.layer_norm,
        activation="relu", 
        layer_num=args.layer_num,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    device = utils.get_device(args)
    model = model.to(device)

    if args.test:
        test(model, test_data, num_users, num_nodes, mode="test", model_name = model_state_file)
    else:
        train_and_validate(args, model, train_data, eval_data, test_data, graphs, num_nodes, 
                           num_users, model_state_file)

    sys.exit()



