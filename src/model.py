from collections.abc import Sequence
import torch
from torch import nn, autograd
from torch.nn import functional as F
import layers
import predict

class PET(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_nodes, num_relation, message_func="transe",
                 short_cut=False, layer_norm=False, activation="relu", num_mlp_layer=2, layer_num=2, num_heads=2, dropout=0.3):
        super(PET,self).__init__()

        self.dims = [input_dim] + list(hidden_dims)
        self.num_nodes = num_nodes
        self.num_relation = num_relation  # reverse rel type should be added
        self.short_cut = short_cut  # whether to use residual connections between layers
        
        # user relation layer
        self.query = nn.Embedding(self.num_relation, input_dim)
        # inital poi embedding
        self.poi = nn.Embedding(self.num_nodes, input_dim)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1): # num of hidden layers
            self.layers.append(layers.PathAggNet(self.dims[i], self.dims[i + 1], self.num_relation,
                                                                self.dims[0], message_func, layer_norm,
                                                                activation))

        self.feature_dim = hidden_dims[-1] + input_dim + input_dim

        # POI temporal gate
        self.gate_weight = nn.Linear(input_dim, input_dim)

        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(self.feature_dim, self.feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(self.feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)
        self.user_dict = {}

        # transformer encoder
        self.layer_num = layer_num
        self.num_heads = num_heads
        self.dropout = dropout
        self.sequence = predict.sequence(self.dims[0]*2, self.num_heads, self.layer_num, self.dropout)

        #Used to obtain the time for each trajectory flow graph.
        self.g_time_embedding = nn.Embedding(48, input_dim)
        self.g_linear = nn.Linear(input_dim, input_dim)


    def memory_update(self, each_snap_g, h_index, query, r_index, first_flag, last_stat):
        batch_size = len(h_index)


        # initialize all pairs states as zeros in memory
        initial_stat = torch.zeros(batch_size, each_snap_g.num_nodes(), self.dims[0], device=h_index.device)
        
        # Obtain the index of each edge and its corresponding embedding
        poi_node = torch.cat((each_snap_g.edges()[0], each_snap_g.edges()[1]),0)
        poi = self.poi(poi_node)
        poi_index = poi_node.unsqueeze(-1).expand_as(poi)


        # Initial first POI embedding of first trajectory flow graph
        initial_stat.scatter_add_(1, poi_index.unsqueeze(0), poi.unsqueeze(0))
        # temporal gate
        gate_weight = torch.sigmoid(self.gate_weight(last_stat))
        initial_stat = first_flag*initial_stat + (1-first_flag)*(initial_stat*gate_weight + (1-gate_weight)*last_stat)
        size = (each_snap_g.num_nodes(), each_snap_g.num_nodes())
        layer_input = initial_stat

        for layer in self.layers:
            # w-layers iteration
            hidden = layer(layer_input, query, initial_stat, torch.stack(each_snap_g.edges()), each_snap_g.edata['type'], size, r_index, edge_weight = None)
            if self.short_cut and hidden.shape == layer_input.shape:
                # shortcut setting
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden

    def forward(self, history_g_list, query_triple, history_poi, history_time_encoding, g_time):
        h_index, r_index, t_index = query_triple.unbind(-1)
        shape = h_index.shape
        batch_size = shape[0]
 
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # Generate all user embedding from the center user view
        query = self.query(r_index[:, 0])
        output = torch.zeros(batch_size, history_g_list[-1].num_nodes(), self.dims[0], device=h_index.device) 
        for ind, each_snap_g in enumerate(history_g_list):
            output = self.memory_update(each_snap_g, h_index[:, 0], query, r_index, first_flag=(ind==0), last_stat = output) 
            #Dynamically update the user embedding
            query = self.g_linear(query + self.g_time_embedding(g_time[ind].to(h_index.device)))           
        feature = output

        # cat center user relation embeddings for enhancing the query processing
        origin_r_emb = query.unsqueeze(1).expand(-1, history_g_list[-1].num_nodes(), -1)
        final_feature = torch.cat([feature, origin_r_emb], dim=-1)

        # # forward Transformer module
        history_poi_index = history_poi.unsqueeze(-1).expand(-1, -1, final_feature.shape[-1])
        feature_input_poi = final_feature.gather(1, history_poi_index)


        # transformer encoder
        feature_label = self.sequence(feature_input_poi, history_time_encoding)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature_t = feature.gather(1, index)
        feature_label = feature_label.expand(-1, feature_t.shape[-2], -1)
        feature_t = torch.cat([feature_t, feature_label], dim=-1)

        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature_t).squeeze(-1)

        return score.view(shape)

    def get_loss(self, args, pred):
        
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        neg_weight = torch.ones_like(pred)
        if args.adversarial_temperature > 0:
            with torch.no_grad():
                neg_weight[:, 1:] = F.softmax(pred[:, 1:] / args.adversarial_temperature, dim=-1)
        else:
            neg_weight[:, 1:] = 1 / args.negative_num
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()

        tmp = torch.mm(self.query.weight, self.query.weight.permute(1, 0))
        orthogonal_regularizer = torch.norm(tmp - 1 * torch.diag(torch.ones(self.num_relation, device=pred.device)), 2)

        loss = loss + orthogonal_regularizer
        return loss
