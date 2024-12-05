import numpy as np
import torch
import dgl
from tqdm import tqdm
from collections import defaultdict
from functools import reduce
import random
import os
from torch import distributed as dist


def set_rand_seed(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # Disable hash randomization
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def get_device(args):
    if args.gpus:
        device = torch.device(args.gpus[get_rank()])
    else:
        device = torch.device("cpu")
    return device


def create_working_directory(args):
    file_name = "working_dir.tmp"
    world_size = get_world_size()
    if args.gpus is not None and len(args.gpus) != world_size:
        error_msg = "World size is %d but found %d GPUs in the argument"
        if world_size == 1:
            error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
        raise ValueError(error_msg % (world_size, len(args.gpus)))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    working_dir = os.path.join("../model", args.dataset)

    # synchronize working directory
    if get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
    synchronize()
    if get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    synchronize()
    if get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def edge_match(edge_index, query_index):
    
    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    # scale = scale[-1] // scale
    scale = torch.div(scale[-1], scale, rounding_mode='floor')

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start # matched if num_match==1

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match

def negative_sampling(all_data, batch_data, num_negative, num_nodes, num_rels, strict=True):

    all_data_t = all_data[:,[0,2,1]] #hrt->htr
    batch_t = batch_data[:,[0,2,1]] #hrt->htr
    data = {
        'num_nodes': num_nodes,
        'edge_index': torch.stack([all_data_t[:,0], all_data_t[:,1]]),
        'edge_type': all_data_t[:,2]
    }

    batch_size = len(batch_t)
    pos_h_index, pos_t_index, pos_r_index = batch_t.t()

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask = strict_negative_mask(data, batch_t)
        
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch_t.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]
    else:
        neg_index = torch.randint(data['num_nodes'], (batch_size, num_negative), device=batch_t.device) 
        neg_t_index = neg_index[:batch_size]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size, 1:] = neg_t_index

    return torch.stack([h_index, r_index, t_index], dim=-1) 

def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data['edge_index'][0], data['edge_type']])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data['edge_index'][1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data['num_nodes'], dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    return t_mask


def all_negative(num_nodes, batch):
    pos_h_index, pos_r_index, pos_t_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, num_nodes)
    # generate all negative tails for this batch
    all_index = torch.arange(num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index)
    t_batch = torch.stack([h_index, r_index, t_index], dim=-1)
    # generate all negative heads for this batch
    all_index = torch.arange(num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index)

    return t_batch


def top_k(t_pred, pos_t_index, topk):
    assert topk > 0
    val, index = torch.topk(t_pred, topk, 1)
    hit = 0
    rank = 0.0
    dcg = 0.0
    for i, p in enumerate(index):
        target = pos_t_index[i]
        if target in p:
            hit += 1
            rank_list = list(p)
            rank_index = rank_list.index(target)
            # rank_index is start from 0, so need plus 1
            rank += 1.0 / (rank_index + 1)
            dcg += 1.0 / np.log2(rank_index + 2)
    return hit, rank, dcg


