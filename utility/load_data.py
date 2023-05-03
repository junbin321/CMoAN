
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import networkx as nx

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []
        
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1
        self.print_statistics()
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                        
                    self.train_items[uid] = train_items
                    
                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    
                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        nodes_info,type2nodes = get_node_info(path + '/node_info.txt')

        self.num_nodes = len(nodes_info)

        type_min_id = {}
        for key,value in type2nodes.items():
            type_min_id[key] = min(value)

        for key,value in type_min_id.items():
            print('type:'+str(key)+'min_id:'+str(value))

        original_graph = get_G_from_edges(load_graph_data(path + '/original_graph_weight.txt'),nodes_info)
        uma_motif_graph = get_G_from_edges(load_graph_data(path + '/uma_motif_graph_weight.txt'),nodes_info)
        umd_motif_graph = get_G_from_edges(load_graph_data(path + '/umd_motif_graph_weight.txt'),nodes_info)
        um2t_motif_graph = get_G_from_edges(load_graph_data(path + '/um1t_motif_graph_weight.txt'), nodes_info)

        # 0-user 1-item 2-actor 3-director 4-type
        user_actor_dict = get_graph_neighbor(uma_motif_graph,'0','2',type_min_id[0],type_min_id[2])
        user_director_dict = get_graph_neighbor(umd_motif_graph,'0','3',type_min_id[0],type_min_id[3])
        user_type_dict = get_graph_neighbor(um2t_motif_graph, '0', '4', type_min_id[0], type_min_id[4])
        item_actor_dict = get_graph_neighbor(original_graph, '1', '2', type_min_id[1], type_min_id[2])
        item_director_dict = get_graph_neighbor(original_graph, '1', '3', type_min_id[1], type_min_id[3])
        item_type_dict = get_graph_neighbor(original_graph, '1', '4', type_min_id[1], type_min_id[4])
        item_item_a_dict = get_graph_neighbor(uma_motif_graph, '1', '1', type_min_id[1], type_min_id[1])
        item_item_d_dict = get_graph_neighbor(umd_motif_graph, '1', '1', type_min_id[1], type_min_id[1])
        item_item_t_dict = get_graph_neighbor(um2t_motif_graph, '1', '1', type_min_id[1], type_min_id[1])

        self.n_actor = len(type2nodes[2])
        self.n_director = len(type2nodes[3])
        self.n_type = len(type2nodes[4])

        self.get_graph_R_weighted(self.n_actor,self.n_director,self.n_type,user_actor_dict,user_director_dict,user_type_dict,
                    item_actor_dict,item_director_dict,item_type_dict,
                    item_item_a_dict,item_item_d_dict,item_item_t_dict,original_graph,uma_motif_graph,
                                  umd_motif_graph,um2t_motif_graph,type_min_id)

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)
        
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
            
        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
        except Exception:
            adj_mat=adj_mat
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            
        return adj_mat, norm_adj_mat, mean_adj_mat,pre_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        # prevent memory from overflowing
        for i in range(5):
            adj_mat[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5), self.n_users:] =\
            R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)]
            adj_mat[self.n_users:,int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)] =\
            R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)].T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        
        t2 = time()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp
        
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)
        
        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
        
    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]


        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    
    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u]+self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items
    
        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    
    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state

    def get_train_data_pairs(self,neg_sample_num):

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)

            return neg_items

        train_pair = []
        for user, pos_items in self.train_items.items():
            neg_items = sample_neg_items_for_u(user, neg_sample_num * len(pos_items))
            for pos_item in pos_items:
                for i in range(neg_sample_num):
                    neg_item = rd.choice(neg_items)
                    train_pair.append([user,pos_item,neg_item])
                    neg_items.remove(neg_item)

        rd.shuffle(train_pair)
        train_pair_array = np.array(train_pair)

        users = train_pair_array[:,0]
        pos_items = train_pair_array[:,1]
        neg_items = train_pair_array[:,2]

        return users,pos_items,neg_items

    def get_test_data_pairs(self):

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.test_set[u] and neg_id not in neg_items:
                    if u not in self.train_items[u]:
                        neg_items.append(neg_id)
                    elif neg_id not in self.train_items[u]:
                        neg_items.append(neg_id)
            return neg_items

        test_pair = []
        for user, pos_items in self.test_set.items():
            neg_items = sample_neg_items_for_u(user, len(pos_items))
            for pos_item in pos_items:
                neg_item = rd.choice(neg_items)
                test_pair.append([user,pos_item,neg_item])
                neg_items.remove(neg_item)

        train_pair_array = np.array(test_pair)

        users = train_pair_array[:,0]
        pos_items = train_pair_array[:,1]
        neg_items = train_pair_array[:,2]

        return users,pos_items,neg_items


    def get_graph_R(self,n_actor,n_director,n_type,user_actor_dict,user_director_dict,user_type_dict,
                    item_actor_dict,item_director_dict,item_type_dict,
                    item_item_a_dict,item_item_d_dict,item_item_t_dict):

        self.R_u_actor = sp.dok_matrix((self.n_users, n_actor), dtype=np.float32)
        for key, value in user_actor_dict.items():
            for i in value:
                self.R_u_actor[key, i] = 1.

        self.R_u_di = sp.dok_matrix((self.n_users, n_director), dtype=np.float32)
        for key, value in user_director_dict.items():
            for i in value:
                self.R_u_di[key, i] = 1.

        self.R_u_type = sp.dok_matrix((self.n_users, n_type), dtype=np.float32)
        for key, value in user_type_dict.items():
            for i in value:
                self.R_u_type[key, i] = 1.

        self.R_i_actor = sp.dok_matrix((self.n_items, n_actor), dtype=np.float32)
        for key, value in item_actor_dict.items():
            for i in value:
                self.R_i_actor[key, i] = 1.

        self.R_i_dir = sp.dok_matrix((self.n_items, n_director), dtype=np.float32)
        for key, value in item_director_dict.items():
            for i in value:
                self.R_i_dir[key, i] = 1.

        self.R_i_type = sp.dok_matrix((self.n_items, n_type), dtype=np.float32)
        for key, value in item_type_dict.items():
            for i in value:
                self.R_i_type[key, i] = 1.

        self.R_i_iactor = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        for key, value in item_item_a_dict.items():
            for i in value:
                self.R_i_iactor[key, i] = 1.

        self.R_i_idir = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        for key, value in item_item_d_dict.items():
            for i in value:
                self.R_i_idir[key, i] = 1.

        self.R_i_itype = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        for key, value in item_item_t_dict.items():
            for i in value:
                self.R_i_itype[key, i] = 1.

    def get_graph_R_weighted(self,n_actor,n_director,n_type,user_actor_dict,user_director_dict,user_type_dict,
                    item_actor_dict,item_director_dict,item_type_dict,
                    item_item_a_dict,item_item_d_dict,item_item_t_dict,original_graph,uma_motif_graph,
                                  umd_motif_graph,um2t_motif_graph,type_min_id):

        self.R_u_actor = sp.dok_matrix((self.n_users, n_actor), dtype=np.float32)
        for key, value in user_actor_dict.items():
            for i in value:
                self.R_u_actor[key, i] = uma_motif_graph[key+type_min_id[0]][i+type_min_id[2]]['weight']

        self.R_u_di = sp.dok_matrix((self.n_users, n_director), dtype=np.float32)
        for key, value in user_director_dict.items():
            for i in value:
                self.R_u_di[key, i] = umd_motif_graph[key+type_min_id[0]][i+type_min_id[3]]['weight']

        self.R_u_type = sp.dok_matrix((self.n_users, n_type), dtype=np.float32)
        for key, value in user_type_dict.items():
            for i in value:
                self.R_u_type[key, i] = um2t_motif_graph[key+type_min_id[0]][i+type_min_id[4]]['weight']

        self.R_i_actor = sp.dok_matrix((self.n_items, n_actor), dtype=np.float32)
        for key, value in item_actor_dict.items():
            for i in value:
                self.R_i_actor[key, i] = original_graph[key+type_min_id[1]][i+type_min_id[2]]['weight']

        self.R_i_dir = sp.dok_matrix((self.n_items, n_director), dtype=np.float32)
        for key, value in item_director_dict.items():
            for i in value:
                self.R_i_dir[key, i] = original_graph[key+type_min_id[1]][i+type_min_id[3]]['weight']

        self.R_i_type = sp.dok_matrix((self.n_items, n_type), dtype=np.float32)
        for key, value in item_type_dict.items():
            for i in value:
                self.R_i_type[key, i] = original_graph[key+type_min_id[1]][i+type_min_id[4]]['weight']

        self.R_i_iactor = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        for key, value in item_item_a_dict.items():
            for i in value:
                self.R_i_iactor[key, i] = uma_motif_graph[key+type_min_id[1]][i+type_min_id[1]]['weight']

        self.R_i_idir = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        for key, value in item_item_d_dict.items():
            for i in value:
                self.R_i_idir[key, i] = umd_motif_graph[key+type_min_id[1]][i+type_min_id[1]]['weight']

        self.R_i_itype = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        for key, value in item_item_t_dict.items():
            for i in value:
                self.R_i_itype[key, i] = um2t_motif_graph[key+type_min_id[1]][i+type_min_id[1]]['weight']


    def create_graph_adj_mat(self, n_nodes1, n_nodes2, R,flag=0):
        t1 = time()
        R = R.tolil()
        if flag==0:
            # prevent memory from overflowing
            adj_mat = sp.dok_matrix((n_nodes1 + n_nodes2, n_nodes1 + n_nodes2), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            for i in range(5):
                adj_mat[int(n_nodes1 * i / 5.0):int(n_nodes1 * (i + 1.0) / 5), n_nodes1:] = \
                    R[int(n_nodes1 * i / 5.0):int(n_nodes1 * (i + 1.0) / 5)]
                adj_mat[n_nodes1:, int(n_nodes1 * i / 5.0):int(n_nodes1 * (i + 1.0) / 5)] = \
                    R[int(n_nodes1 * i / 5.0):int(n_nodes1 * (i + 1.0) / 5)].T
        elif flag==1:
            adj_mat = sp.dok_matrix((n_nodes1, n_nodes1), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            for i in range(5):
                adj_mat[int(n_nodes1 * i / 5.0):int(n_nodes1 * (i + 1.0) / 5), :n_nodes1] = \
                    R[int(n_nodes1 * i / 5.0):int(n_nodes1 * (i + 1.0) / 5)]

        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)

        adj_mat = adj_mat.tocsr()
        adj_mat = adj_mat
        norm_adj_mat = norm_adj_mat.tocsr()
        mean_adj_mat = mean_adj_mat.tocsr()

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()

        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat_inv)
        print('generate pre adjacency matrix.')
        pre_adj_mat = norm_adj.tocsr()

        return adj_mat, norm_adj_mat, mean_adj_mat, pre_adj_mat

    def create_graph_adj_mat1(self,n_nodes1,n_nodes2,R1,R2=None):
        t1 = time()
        adj_mat = sp.dok_matrix((n_nodes1 + n_nodes2, n_nodes1 + n_nodes2), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R1 = R1.tolil()
        # prevent memory from overflowing
        for i in range(5):
            adj_mat[int(n_nodes1 * i / 5.0):int(n_nodes1 * (i + 1.0) / 5), n_nodes1:] = \
                R1[int(n_nodes1 * i / 5.0):int(n_nodes1 * (i + 1.0) / 5)]
            adj_mat[n_nodes1:, int(n_nodes1 * i / 5.0):int(n_nodes1 * (i + 1.0) / 5)] = \
                R1[int(n_nodes1 * i / 5.0):int(n_nodes1 * (i + 1.0) / 5)].T

        if R2:
            R2 = R2.tolil()
            for i in range(5):
                adj_mat[int(n_nodes1 * i / 5.0):int(n_nodes1 * (i + 1.0) / 5), :n_nodes1] = \
                    R2[int(n_nodes1 * i / 5.0):int(n_nodes1 * (i + 1.0) / 5)]

        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)

        adj_mat = adj_mat.tocsr()
        norm_adj_mat = norm_adj_mat.tocsr()
        mean_adj_mat = mean_adj_mat.tocsr()

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()

        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat_inv)
        print('generate pre adjacency matrix.')
        pre_adj_mat = norm_adj.tocsr()

        return adj_mat, norm_adj_mat, mean_adj_mat, pre_adj_mat


def load_graph_data(f_name):
    print('We are loading graph data from:', f_name)
    all_edges = list()
    with open(f_name, 'r') as f:
        for line in f.readlines():
            words = line.strip().split('\t')
            x, y = int(words[0]), int(words[1])
            all_edges.append((x, y))
    print('Total node pairs: ' + str(len(all_edges)))
    return all_edges

def get_node_info(f_name):
    node_info = {}
    type2nodes = {}

    with open(f_name,'r') as f:
        for line in f.readlines():
            array = line.strip().split('\t')
            node_idx = int(array[0])
            node_type = int(array[1])
            node_info[node_idx] = node_type
            if node_type not in type2nodes:
                type2nodes[node_type] = []
            type2nodes[node_type].append(node_idx)

    print('node number is ',len(node_info))

    return node_info,type2nodes


def get_G_from_edges(edges,node_info):

    tmp_G = nx.Graph()
    for key,value in node_info.items():
        tmp_G.add_node(key)
        tmp_G.nodes[key]['type'] = value

    edge_dict = dict()
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        x = int(edge_key.split('_')[0])
        y = int(edge_key.split('_')[1])
        tmp_G.add_edge(x, y)
        tmp_G[x][y]['weight'] = weight
    return tmp_G

def get_graph_neighbor(graph,source_type,end_type,source_type_min_id,end_type_min_id):
    graph_node = nx.nodes(graph)
    source_nodes = []
    for i in graph_node:
        if graph.nodes[i]['type'] == int(source_type):
            source_nodes.append(i)
    neighbor_dic = {}
    for source_node in source_nodes:
        node_neighbors_type = []
        node_neighbors = nx.all_neighbors(graph, source_node)
        for neighbor in node_neighbors:
            if graph.nodes[neighbor]['type'] == int(end_type):
                node_neighbors_type.append(neighbor-end_type_min_id)
        neighbor_dic[source_node-source_type_min_id] = node_neighbors_type

    return neighbor_dic

