from tqdm import tqdm
import json
import math
import os
import json
import pandas as pd
import numpy as np
import dgl
import torch
from datetime import datetime, timedelta
from collections import Counter
from geopy import distance


class DataEncoder(object):
    def __init__(self):
        self.num_nodes = 1950
        self.num_users = 767

        self.train_rate = 0.7
        self.eval_rate = 0.1

        # POI dictory
        self.d_poi = {}
        # the first value of POI in dictory
        self.loc_id = 0
        # the first value of POI in dictory
        self.user_id = 0


    def return_num_nodes(self, dataset):
        if dataset == 'PHO':
            self.num_nodes = 1950
        elif dataset == 'NYC':
            self.num_nodes = 15624
        elif dataset == 'TKY':
            self.num_nodes = 24321
        return self.num_nodes
    
    def return_num_users(self, dataset):
        if dataset == 'PHO':
            self.num_users = 767
        elif dataset == 'NYC':
            self.num_users = 1083
        elif dataset == 'TKY':
            self.num_users = 2293
        return self.num_users

    def get_data(self, dataset):
        if dataset == 'PHO':
            self.num_nodes = 1950
        elif dataset == 'NYC':
            self.num_nodes = 15624
        elif dataset == 'TKY':
            self.num_nodes = 24321
        data_path = f'../../data/{dataset}.csv'

        # Read the raw dataset
        df = pd.read_csv(os.path.join(data_path))

        # 1.cut data for each user
        cut_data = self.cutter_filter(df)

        # 2.get the each snapshot and build temporal trajectory flow graphs
        df_graph = self.build_dfgraph(cut_data)
        graphs = self.build_graphs(df_graph)

        # 3.encode the cut data
        encoded_data = self.encode_traj(cut_data)

        # 4.divide the dataset into train_data, eval_data and test_data
        train_data, eval_data, test_data = self.divide_data(df, encoded_data)

        return graphs, train_data, eval_data, test_data

    def encode(self, trajectories):
        # Record the encoded trajectory.
        encoded_trajectories = []
        for traj in trajectories:
            current_loc = []
            current_day_period = []
            current_utc = []
            current_tim = []
            for point in traj:
                current_loc.append(point[2])
                current_day_period.append(point[3])
                current_utc.append(point[0])
                time_code = self._time_encode(point[0])
                current_tim.append(time_code)
            # Complete the encoding of the current trajectory, then proceed with the formation of the input
            trace = []
            target = current_loc[-1]

            # 0.history_poi
            trace.append(current_loc[:-1]) 
            # 1.history_day_period
            trace.append(current_day_period[:-1])
            # 2.history_poi_utc
            trace.append(current_utc.copy())    
            # 3.history_time_encoding
            trace.append(current_tim[:-1])                                             
            # 4.current_poi
            trace.append(current_loc[-2])
            # 5.user
            trace.append(point[1])
            # 6.target_poi
            trace.append(target)
            # 7.target_day_period
            trace.append(current_day_period[-1])
            # 8.current_poi_utc
            trace.append(current_utc[-2])
            # 9.target_poi_utc
            trace.append(current_utc[-1])

            encoded_trajectories.append(trace)
        return encoded_trajectories

    def encode_traj(self, data):
        encoded_data = {}
        for uid in tqdm(data):
            encoded_data[uid] = self.encode(data[uid])
        return encoded_data
    
    def divide_data(self, df, data):
        train_data = []
        eval_data = []
        test_data = []

        train_time_divide = df['time'][math.ceil(len(df) * self.train_rate)]
        eval_time_divide = df['time'][math.ceil(len(df) * (self.train_rate + self.eval_rate))]

        user_set = data.keys()
        for uid in tqdm(user_set):
            encoded_trajectories = data[uid]
            for traj in encoded_trajectories:
                # traj[9] = target_poi_utc
                if traj[9] < train_time_divide:
                    train_data.append(traj)
                elif train_time_divide <= traj[9] < eval_time_divide:
                    eval_data.append(traj)
                else:
                    test_data.append(traj)
        print(len(train_data), len(eval_data), len(test_data))

        # Rearrange the order of the training set, validation set, and test set to be sorted by target_time.
        name = ['history_poi', 'history_day_period', 'history_poi_utc', 'history_time_encoding','current_poi', 'user', 'target_poi', 'target_day_period', 'current_poi_utc', 'target_poi_utc',]
        pd_train = pd.DataFrame(data = train_data,index = None,columns = name)
        pd_train = pd_train.sort_values(by='target_poi_utc')
        pd_val = pd.DataFrame(data = eval_data,index = None,columns = name)
        pd_val = pd_val.sort_values(by='target_poi_utc')
        pd_test = pd.DataFrame(data = test_data,index = None,columns = name)
        pd_test = pd_test.sort_values(by='target_poi_utc')
        # Convert to list format.
        train_data = np.array(pd_train).tolist()
        eval_data = np.array(pd_val).tolist()
        test_data = np.array(pd_test).tolist()

        return train_data, eval_data, test_data

    def cutter_filter(self, traj):
        # Calculate the time required for the training set corresponding to the given ratio
        train_time = traj['time'][math.ceil(len(traj) * self.train_rate)]
        # Calculate the time required for the validation set corresponding to the given ratio
        eval_time = traj['time'][math.ceil(len(traj) * (self.train_rate + self.eval_rate))]
        # User ID
        user_set = pd.unique(traj['entity_id'])
        # Cut data sample
        cut_data = {}
        # Each user has at least min_session_len number of trajectories for training
        min_session_len = 3
        for uid in tqdm(user_set):
            usr_traj = traj[traj['entity_id'] == uid].values.tolist()
            sessions = []
            session_train = [] # training sample
            session_eval = [] # evaluation sample
            session_test = [] # testing sample
            for row in usr_traj:
                now_time = row[0]
                if now_time < train_time:
                    session_train.append(row)
                    # Each trajectory must contain at least 24 check-ins
                    if len(session_train) >= 24:
                        sessions.append(session_train) 
                        session_train = []
                elif train_time <= now_time < eval_time:
                    session_eval.append(row)
                elif eval_time <= now_time:
                    session_test.append(row)
            if len(session_train) >= min_session_len:   
                sessions.append(session_train) 
            if len(session_eval) >= min_session_len:   
                sessions.append(session_eval) 
            if len(session_test) >= min_session_len:   
                sessions.append(session_test) 
            cut_data[str(uid)] = sessions
        # Reindex the data
        for uid in cut_data:
            data = cut_data[uid]
            user_id = self.user_id
            self.user_id += 1
            for index_traj, traj in enumerate(data):
                for index_point, point in enumerate(traj):
                    loc = point[2]
                    if loc not in self.d_poi:
                        self.d_poi[loc] = self.loc_id
                        self.loc_id += 1
                    cut_data[uid][index_traj][index_point][1] = user_id
                    cut_data[uid][index_traj][index_point][2] = self.d_poi[loc]

        return cut_data

    def convertdata(self, data):
        list_all = []
        users = list(set(data['entity_id'].to_list()))
        for user_id in users:
            user_df = data[data['entity_id'] == user_id]
            previous_poi_id = None
            previous_poi_utc = None
            for i, row in user_df.iterrows():
                target_poi_id = row['location']
                target_day_period = row['timediff'] 
                target_poi_utc =  row['time'] 
                if (previous_poi_id == None):
                    previous_poi_id = target_poi_id
                    previous_poi_utc = target_poi_utc
                else:
                    list_all.append([previous_poi_id, user_id, target_poi_id, previous_poi_utc, target_poi_utc, target_day_period])
                previous_poi_id = target_poi_id
                previous_poi_utc = target_poi_utc
        sorted_list = sorted(list_all, key = lambda x:x[4])
        return sorted_list
    
    def build_graphs(self, data):
        # Transform the dataset into the format of [previous POI id, User id, next POI id, previous time, next time, the index of temporal period of next time].
        data = self.convertdata(data)
        # Get the snapshot of each temporal period
        sp_all = self.split_by_time(data)
        # temporal trajectory flow graphs
        g_list = {}
        # get trajectory flow graph of each temporal period
        for index in tqdm(sp_all):
            g_list[index] = self.build_graph(sp_all[index])
        
        return g_list

    def build_graph(self, snapshot):
        src, rel, dst, src_utc, dst_utc = snapshot.transpose()

        # Convert the data type.
        src = src.astype(int)
        dst = dst.astype(int)
        rel = rel.astype(int)

        # construct trajectory flow graph
        g = dgl.graph(([],[]))
        g.add_nodes(self.num_nodes)
        g.add_edges(src, dst)
        g.edata['type'] = torch.LongTensor(rel)

        return g

    def split_by_time(self, data):
        # 'data' type: [previous POI id, User id, next POI id, previous time, next time, the index of temporal period of next time]
        snapshot_list = {}
        snapshot = []
        snapshots_num = 0
        latest_t = 0
        for i in range(len(data)):
            t = data[i][5]
            train = data[i][:5]
            if latest_t != t: 
                if len(snapshot):
                    snapshot_list[latest_t] = np.array(snapshot).copy()
                    snapshots_num += 1
                latest_t = t

                snapshot = []
            if train not in snapshot:
                snapshot.append(train)     
        # the last snapshot
        if len(snapshot) > 0:
            snapshot_list[latest_t] = np.array(snapshot).copy()
            snapshots_num += 1

        # Save the snapshot data of each temporal period
        snapshot_file_name = '../../data/graphs/snapshot.json'
        snapshot_list_json = snapshot_list.copy()
        for key in snapshot_list_json.keys():
            snapshot_list_json[key] = snapshot_list_json[key].tolist()
        with open(snapshot_file_name, 'w', encoding='UTF-8') as f:
            f.write(str(snapshot_list_json))

        return snapshot_list

    def build_dfgraph(self, cut_data):
        cut_data_values = []
        for value in cut_data.values():
            cut_data_values.extend(value)

        data = []
        for index1 in range(len(cut_data_values)):
            for index2 in range(len(cut_data_values[index1])):
                data.append(cut_data_values[index1][index2])

        name = ['time', 'entity_id', 'location', 'timediff']
        dfgraph = pd.DataFrame(data = data,index = None,columns = name)
        # Sort the dfgraph by time.
        dfgraph = dfgraph.sort_values(by='time')
        # Reindex
        dfgraph = dfgraph.reset_index(drop=True)

        return dfgraph
    
    def _time_encode(self, time):
        time = parse_time(time)
        if time.weekday() in [0, 1, 2, 3, 4]:
            return time.hour
        else:
            return time.hour + 24


def parse_time(time_in, timezone_offset_in_minute=0):
    # Convert the time in time_format from the JSON to local datetime.
    date = datetime.strptime(time_in, '%Y-%m-%dT%H:%M:%SZ')
    return date + timedelta(minutes=timezone_offset_in_minute)
