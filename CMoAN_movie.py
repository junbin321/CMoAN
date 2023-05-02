
import os
import sys
import threading
import tensorflow as tf
from tensorflow.python.client import device_lib
from utility.helper import *
from utility.batch_test import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']


class CMoAN_movie(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.attention_dim =16
        self.node_attention_dim = 8

        self.model_type = 'CMoAN'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_actor = data_config['n_actor']
        self.n_director = data_config['n_director']
        self.n_type = data_config['n_type']

        self.n_fold = 10
        self.ui_pre_adj = data_config['ui_norm_adj']
        self.u_actor_pre_adj = data_config['u_actor_pre_adj_mat']
        self.u_di_pre_adj = data_config['u_di_pre_adj_mat']
        self.u_type_pre_adj = data_config['u_type_pre_adj_mat']
        self.i_actor_pre_adj = data_config['i_actor_pre_adj_mat']
        self.i_di_pre_adj = data_config['i_di_pre_adj_mat']
        self.i_type_pre_adj = data_config['i_type_pre_adj_mat']
        self.i_ia_pre_adj = data_config['i_ia_pre_adj_mat']
        self.i_idir_pre_adj = data_config['i_idir_pre_adj_mat']
        self.i_itype_pre_adj = data_config['i_itype_pre_adj_mat']


        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.log_dir = self.create_model_str()
        self.verbose = args.verbose
        self.Ks = eval(args.Ks)

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])
        with tf.name_scope('TRAIN_LOSS'):
            self.train_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_loss', self.train_loss)
            self.train_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_mf_loss', self.train_mf_loss)
            self.train_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_emb_loss', self.train_emb_loss)
            self.train_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_reg_loss', self.train_reg_loss)
        self.merged_train_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_LOSS'))

        with tf.name_scope('TRAIN_ACC'):
            self.train_rec_first = tf.placeholder(tf.float32)
            # record for top(Ks[0])
            tf.summary.scalar('train_rec_first', self.train_rec_first)
            self.train_rec_last = tf.placeholder(tf.float32)
            # record for top(Ks[-1])
            tf.summary.scalar('train_rec_last', self.train_rec_last)
            self.train_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg_first', self.train_ndcg_first)
            self.train_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg_last', self.train_ndcg_last)
        self.merged_train_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_ACC'))

        with tf.name_scope('TEST_LOSS'):
            self.test_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_loss', self.test_loss)
            self.test_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_mf_loss', self.test_mf_loss)
            self.test_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_emb_loss', self.test_emb_loss)
            self.test_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_reg_loss', self.test_reg_loss)
        self.merged_test_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_LOSS'))

        with tf.name_scope('TEST_ACC'):
            self.test_rec_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec_first', self.test_rec_first)
            self.test_rec_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec_last', self.test_rec_last)
            self.test_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_first', self.test_ndcg_first)
            self.test_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_last', self.test_ndcg_last)
        self.merged_test_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_ACC'))
        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        self.ua_embeddings, self.ia_embeddings = self._create_HINMotifGCN_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False,
                                       transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def create_model_str(self):
        log_dir = '/' + self.alg_type + '/layers_' + str(self.n_layers) + '/dim_' + str(self.emb_dim)
        log_dir += '/' + args.dataset + '/lr_' + str(self.lr) + '/reg_' + str(self.decay)
        return log_dir

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                        name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='item_embedding')
            all_weights['actor_embedding'] = tf.Variable(initializer([self.n_actor, self.emb_dim]),
                                                        name='actor_embedding')
            all_weights['director_embedding'] = tf.Variable(initializer([self.n_director, self.emb_dim]),
                                                        name='director_embedding')
            all_weights['type_embedding'] = tf.Variable(initializer([self.n_type, self.emb_dim]),
                                                        name='type_embedding')

            all_weights['item_node_attention_trans_w'] = tf.Variable(
                initializer([self.emb_dim * 2, self.node_attention_dim]),
                name='item_node_attention_trans_w')

            all_weights['item_node_attention_vector'] = tf.Variable(initializer([1, self.node_attention_dim]),
                                                                    name='item_node_attention_vector')

            all_weights['actor_node_attention_trans_w'] = tf.Variable(
                initializer([self.emb_dim * 2, self.node_attention_dim]),
                name='actor_node_attention_trans_w')

            all_weights['actor_node_attention_vector'] = tf.Variable(initializer([1, self.node_attention_dim]),
                                                                  name='actor_node_attention_vector')

            all_weights['dir_node_attention_trans_w'] = tf.Variable(
                initializer([self.emb_dim * 2, self.node_attention_dim]),
                name='dir_node_attention_trans_w')

            all_weights['dir_node_attention_vector'] = tf.Variable(initializer([1, self.node_attention_dim]),
                                                                  name='dir_node_attention_vector')

            all_weights['type_node_attention_trans_w'] = tf.Variable(
                initializer([self.emb_dim * 2, self.node_attention_dim]),
                name='type_node_attention_trans_w')

            all_weights['type_node_attention_vector'] = tf.Variable(initializer([1, self.node_attention_dim]),
                                                                  name='type_node_attention_vector')

            all_weights['user_attention_vector'] = tf.Variable(initializer([self.n_users, self.attention_dim]),
                                                               name='user_attention_vector')

            all_weights['item_attention_vector'] = tf.Variable(initializer([self.n_items, self.attention_dim]),
                                                               name='item_attention_vector')

            all_weights['ucf_attention_trans_w'] = tf.Variable(initializer([self.emb_dim, self.attention_dim]),
                                                               name='ucf_attention_trans_w')

            all_weights['ucf_attention_trans_b'] = tf.Variable(initializer([1,self.attention_dim]),
                                                               name='ucf_attention_trans_b')

            all_weights['ua_attention_trans_w'] = tf.Variable(initializer([self.emb_dim, self.attention_dim]),
                                                               name='ua_attention_trans_w')

            all_weights['ua_attention_trans_b'] = tf.Variable(initializer([1,self.attention_dim]),
                                                               name='ua_attention_trans_b')

            all_weights['ud_attention_trans_w'] = tf.Variable(initializer([self.emb_dim, self.attention_dim]),
                                                               name='ud_attention_trans_w')

            all_weights['ud_attention_trans_b'] = tf.Variable(initializer([1,self.attention_dim]),
                                                               name='ud_attention_trans_b')

            all_weights['ut_attention_trans_w'] = tf.Variable(initializer([self.emb_dim, self.attention_dim]),
                                                               name='ut_attention_trans_w')

            all_weights['ut_attention_trans_b'] = tf.Variable(initializer([1,self.attention_dim]),
                                                               name='ut_attention_trans_b')

            all_weights['icf_attention_trans_w'] = tf.Variable(initializer([self.emb_dim, self.attention_dim]),
                                                               name='icf_attention_trans_w')

            all_weights['icf_attention_trans_b'] = tf.Variable(initializer([1,self.attention_dim]),
                                                               name='icf_attention_trans_b')

            all_weights['ia_attention_trans_w'] = tf.Variable(initializer([self.emb_dim, self.attention_dim]),
                                                               name='ia_attention_trans_w')

            all_weights['ia_attention_trans_b'] = tf.Variable(initializer([1,self.attention_dim]),
                                                               name='ia_attention_trans_b')

            all_weights['id_attention_trans_w'] = tf.Variable(initializer([self.emb_dim, self.attention_dim]),
                                                               name='id_attention_trans_w')

            all_weights['id_attention_trans_b'] = tf.Variable(initializer([1,self.attention_dim]),
                                                               name='id_attention_trans_b')

            all_weights['it_attention_trans_w'] = tf.Variable(initializer([self.emb_dim, self.attention_dim]),
                                                               name='it_attention_trans_w')

            all_weights['it_attention_trans_b'] = tf.Variable(initializer([1,self.attention_dim]),
                                                               name='it_attention_trans_b')

            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            all_weights['actor_embedding'] = tf.Variable(initial_value=self.pretrain_data['actor_embed'], trainable=True,
                                                        name='actor_embedding', dtype=tf.float32)
            all_weights['director_embedding'] = tf.Variable(initial_value=self.pretrain_data['director_embed'], trainable=True,
                                                        name='director_embedding', dtype=tf.float32)
            all_weights['type_embedding'] = tf.Variable(initial_value=self.pretrain_data['type_embed'], trainable=True,
                                                        name='type_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        return all_weights

    def _split_A_hat_general(self, X,n_nodes):
        A_fold_hat = []

        fold_len = n_nodes // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = n_nodes
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat


    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    '''
        forward
    '''
    def _create_HINMotifGCN_embed(self):

        users_all_embeddings = []
        items_all_embeddings = []
        user_cos_score_ls = []
        item_cos_score_ls = []

        # cf aggregate
        users_cf_all_embeddings = [self.weights['user_embedding']]
        items_cf_all_embeddings = [self.weights['item_embedding']]
        ui_layer_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        for k in range(0, self.n_layers):
            ui_A_fold_hat = self._split_A_hat_general(self.ui_pre_adj, self.n_users + self.n_items)
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(ui_A_fold_hat[f], ui_layer_embeddings))

            cf_side_embeddings = tf.concat(temp_embed, 0)
            ui_layer_embeddings = cf_side_embeddings
            u_cf_embeddings, i_cf_embeddings = tf.split(cf_side_embeddings, [self.n_users, self.n_items], 0)# 第二个参数：传入的是一个向量（这里向量各个元素的和要跟原本这个维度的数值相等）就根据这个向量有几个元素分为几项）

            users_cf_all_embeddings += [u_cf_embeddings]
            items_cf_all_embeddings += [i_cf_embeddings]

        # cf attention
        users_cf_all_embeddings = tf.stack(users_cf_all_embeddings, axis=0)
        users_cf_final_embeddings = tf.reduce_mean(users_cf_all_embeddings, axis=0, keepdims=False)  # n_user,dim
        user_attention_vector = self.weights['user_attention_vector']  # n_user,dim

        users_cf_final_embeddings_trans = tf.matmul(users_cf_final_embeddings,
                                                    self.weights['ucf_attention_trans_w']) + self.weights[
                                              'ucf_attention_trans_b']
        users_cf_final_embeddings_trans = tf.nn.tanh(users_cf_final_embeddings_trans)

        user_cf_cos_simi = self.cosine_distance(users_cf_final_embeddings_trans, user_attention_vector)  # n_user,
        user_cos_score_ls += [user_cf_cos_simi]
        users_all_embeddings += [users_cf_final_embeddings]

        items_cf_all_embeddings = tf.stack(items_cf_all_embeddings, axis=0)
        items_cf_final_embeddings = tf.reduce_mean(items_cf_all_embeddings, axis=0, keepdims=False)  # n_item,dim
        item_attention_vector = self.weights['item_attention_vector']  # n_item,dim
        items_cf_final_embeddings_trans = tf.matmul(items_cf_final_embeddings,
                                                    self.weights['icf_attention_trans_w']) + self.weights[
                                              'icf_attention_trans_b']
        items_cf_final_embeddings_trans = tf.nn.tanh(items_cf_final_embeddings_trans)

        item_cf_cos_simi = self.cosine_distance(items_cf_final_embeddings_trans, item_attention_vector)  # n_item,
        item_cos_score_ls += [item_cf_cos_simi]

        items_all_embeddings += [items_cf_final_embeddings]

        # u_i_actor aggregate
        users_embeddings = self.weights['user_embedding']
        items_embeddings = self.weights['item_embedding']
        actor_embeddings = self.weights['actor_embedding']

        users_actor_all_embeddings = [self.weights['user_embedding']]
        items_actor_all_embeddings = [self.weights['item_embedding']]

        u_a_layer_embeddings = tf.concat([users_embeddings, actor_embeddings], axis=0)
        i_a_layer_embeddings = tf.concat([items_embeddings, actor_embeddings], axis=0)
        ii_a_layer_embeddings = items_embeddings

        for k in range(0, self.n_layers):

            item_node_attenton_score_ls = []
            a_node_attenton_score_ls = []
            item_type_agg_all_embeddings = []
            a_type_agg_all_embeddings = []

            u_a_A_fold_hat = self._split_A_hat_general(self.u_actor_pre_adj, self.n_users + self.n_actor)
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(u_a_A_fold_hat[f], u_a_layer_embeddings))

            ua_side_embeddings = tf.concat(temp_embed, 0)
            u_ua_embeddings, a_ua_embeddings = tf.split(ua_side_embeddings, [self.n_users, self.n_actor], 0)

            # user ua attention

            # actor ua attention
            a_ua_self_atten_embed = tf.concat([actor_embeddings, a_ua_embeddings], 1)  # a_num,2dim

            a_ua_self_atten_embed_trans = tf.matmul(a_ua_self_atten_embed,
                                                      self.weights['actor_node_attention_trans_w'])
            a_ua_cos_simi = tf.nn.leaky_relu(tf.squeeze(tf.matmul(a_ua_self_atten_embed_trans,
                                                                    tf.transpose(
                                                                        self.weights['actor_node_attention_vector'],
                                                                        perm=[1, 0]))))  # a_num,

            a_node_attenton_score_ls += [a_ua_cos_simi]
            a_type_agg_all_embeddings += [a_ua_embeddings]

            # i_actor_aggregate
            i_a_A_fold_hat = self._split_A_hat_general(self.i_actor_pre_adj, self.n_items + self.n_actor)
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(i_a_A_fold_hat[f], i_a_layer_embeddings))

            ia_side_embeddings = tf.concat(temp_embed, 0)

            i_ia_embeddings, a_ia_embeddings = tf.split(ia_side_embeddings, [self.n_items, self.n_actor], 0)

            # item ica attention
            i_ia_self_atten_embed = tf.concat([items_embeddings, i_ia_embeddings], 1)  # item_num,2dim

            i_ia_self_atten_embed_trans = tf.matmul(i_ia_self_atten_embed,
                                                     self.weights['item_node_attention_trans_w'])
            i_ia_cos_simi = tf.nn.leaky_relu(tf.squeeze(tf.matmul(i_ia_self_atten_embed_trans,
                                                                   tf.transpose(
                                                                       self.weights['item_node_attention_vector'],
                                                                       perm=[1, 0]))))  # item_num,

            item_node_attenton_score_ls += [i_ia_cos_simi]
            item_type_agg_all_embeddings += [i_ia_embeddings]

            # actor ia attention
            a_ia_self_atten_embed = tf.concat([actor_embeddings, a_ia_embeddings], 1)  # a_num,2dim

            a_ia_self_atten_embed_trans = tf.matmul(a_ia_self_atten_embed,
                                                      self.weights['actor_node_attention_trans_w'])
            a_ia_cos_simi = tf.nn.leaky_relu(tf.squeeze(tf.matmul(a_ia_self_atten_embed_trans,
                                                                    tf.transpose(
                                                                        self.weights['actor_node_attention_vector'],
                                                                        perm=[1, 0]))))  # a_num,

            a_node_attenton_score_ls += [a_ia_cos_simi]
            a_type_agg_all_embeddings += [a_ia_embeddings]

            i_ia_A_fold_hat = self._split_A_hat_general(self.i_ia_pre_adj, self.n_items)
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(i_ia_A_fold_hat[f], ii_a_layer_embeddings))

            i_ia_side_embeddings = tf.concat(temp_embed, 0)

            i_i_ia_embeddings = i_ia_side_embeddings

            # item i_i ia attention
            i_i_ia_self_atten_embed = tf.concat([items_embeddings, i_i_ia_embeddings], 1)  # item_num,2dim

            i_i_ia_self_atten_embed_trans = tf.matmul(i_i_ia_self_atten_embed,
                                                       self.weights['item_node_attention_trans_w'])

            i_i_ia_cos_simi = tf.nn.leaky_relu(tf.squeeze(tf.matmul(i_i_ia_self_atten_embed_trans,
                                                                     tf.transpose(
                                                                         self.weights['item_node_attention_vector'],
                                                                         perm=[1, 0]))))  # item_num,

            item_node_attenton_score_ls += [i_i_ia_cos_simi]
            item_type_agg_all_embeddings += [i_i_ia_embeddings]

            # attention agg
            item_type_agg_all_embeddings = tf.stack(item_type_agg_all_embeddings, 0)  # 2,n_item,dim
            a_type_agg_all_embeddings = tf.stack(a_type_agg_all_embeddings, 0)  # 2,n_actor,dim

            item_node_attenton_score = tf.stack(item_node_attenton_score_ls, 0)  # 2,n_item
            a_node_attenton_score = tf.stack(a_node_attenton_score_ls, 0)  # 2,n_actor

            item_node_attenton_score = tf.nn.softmax(item_node_attenton_score, axis=0)  # 2,n_item

            a_node_attenton_score = tf.nn.softmax(a_node_attenton_score, axis=0)  # 2,n_actor

            item_node_attenton_score = tf.tile(tf.expand_dims(item_node_attenton_score, axis=2),
                                               multiples=[1, 1, self.emb_dim])  # 2,n_item,dim
            a_node_attenton_score = tf.tile(tf.expand_dims(a_node_attenton_score, axis=2),
                                             multiples=[1, 1, self.emb_dim])  # 2,n_actor,dim

            item_type_agg_embeddings = tf.reduce_sum(tf.multiply(item_type_agg_all_embeddings, item_node_attenton_score),
                                                     axis=0, keepdims=False)  # n_item,dim
            a_type_agg_embeddings = tf.reduce_sum(tf.multiply(a_type_agg_all_embeddings, a_node_attenton_score),
                                                   axis=0, keepdims=False)  # n_actor,dim

            users_actor_all_embeddings += [u_ua_embeddings]
            items_actor_all_embeddings += [item_type_agg_embeddings]

            users_embeddings = u_ua_embeddings
            items_embeddings = item_type_agg_embeddings
            actor_embeddings = a_type_agg_embeddings

            u_a_layer_embeddings = tf.concat([users_embeddings, actor_embeddings], axis=0)
            i_a_layer_embeddings = tf.concat([items_embeddings, actor_embeddings], axis=0)
            ii_a_layer_embeddings = items_embeddings

        # u_i_actor attention
        users_actor_all_embeddings = tf.stack(users_actor_all_embeddings, axis=0)
        users_actor_final_embeddings = tf.reduce_mean(users_actor_all_embeddings, axis=0, keepdims=False)  # n_user,dim
        user_attention_vector = self.weights['user_attention_vector']  # n_user,dim
        users_actor_final_embeddings_trans = tf.matmul(users_actor_final_embeddings,
                                                    self.weights['ua_attention_trans_w']) + self.weights[
                                              'ua_attention_trans_b']
        users_actor_final_embeddings_trans = tf.nn.tanh(users_actor_final_embeddings_trans)
        user_actor_cos_simi = self.cosine_distance(users_actor_final_embeddings_trans, user_attention_vector)  # n_user,
        user_cos_score_ls += [user_actor_cos_simi]

        users_all_embeddings += [users_actor_final_embeddings]

        # i_actor attention
        items_a_all_embeddings = tf.stack(items_actor_all_embeddings, axis=0)
        items_a_final_embeddings = tf.reduce_mean(items_a_all_embeddings, axis=0, keepdims=False)  # n_item,dim
        item_attention_vector = self.weights['item_attention_vector']  # n_item,dim
        items_a_final_embeddings_trans = tf.matmul(items_a_final_embeddings,
                                                   self.weights['ia_attention_trans_w']) + self.weights[
                                             'ia_attention_trans_b']
        items_a_final_embeddings_trans = tf.nn.tanh(items_a_final_embeddings_trans)

        item_a_cos_simi = self.cosine_distance(items_a_final_embeddings_trans, item_attention_vector)  # n_item,
        item_cos_score_ls += [item_a_cos_simi]

        items_all_embeddings += [items_a_final_embeddings]

        # u_i_dir aggregate
        users_embeddings = self.weights['user_embedding']
        items_embeddings = self.weights['item_embedding']
        director_embeddings = self.weights['director_embedding']

        users_dir_all_embeddings = [self.weights['user_embedding']]
        items_dir_all_embeddings = [self.weights['item_embedding']]

        udir_layer_embeddings = tf.concat([users_embeddings, director_embeddings], axis=0)
        idir_layer_embeddings = tf.concat([items_embeddings, director_embeddings], axis=0)
        iidir_layer_embeddings = items_embeddings

        for k in range(0, self.n_layers):

            item_node_attenton_score_ls = []
            dir_node_attenton_score_ls = []
            item_type_agg_all_embeddings = []
            dir_type_agg_all_embeddings = []

            u_dir_A_fold_hat = self._split_A_hat_general(self.u_di_pre_adj, self.n_users + self.n_director)
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(u_dir_A_fold_hat[f], udir_layer_embeddings))

            udir_side_embeddings = tf.concat(temp_embed, 0)

            u_udir_embeddings, d_udir_embeddings = tf.split(udir_side_embeddings, [self.n_users, self.n_director], 0)

            # user udir attention

            # director udir attention
            d_udir_self_atten_embed = tf.concat([director_embeddings, d_udir_embeddings], 1)  # dir_num,2dim

            d_udir_self_atten_embed_trans = tf.matmul(d_udir_self_atten_embed,
                                                      self.weights['dir_node_attention_trans_w'])
            d_udir_cos_simi = tf.nn.leaky_relu(tf.squeeze(tf.matmul(d_udir_self_atten_embed_trans,
                                                                    tf.transpose(
                                                                        self.weights['dir_node_attention_vector'],
                                                                        perm=[1, 0]))))  # ca_num,

            dir_node_attenton_score_ls += [d_udir_cos_simi]
            dir_type_agg_all_embeddings += [d_udir_embeddings]

            i_dir_A_fold_hat = self._split_A_hat_general(self.i_di_pre_adj, self.n_items + self.n_director)
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(i_dir_A_fold_hat[f], idir_layer_embeddings))

            idir_side_embeddings = tf.concat(temp_embed, 0)

            i_idir_embeddings, d_idir_embeddings = tf.split(idir_side_embeddings, [self.n_items, self.n_director], 0)

            # item idir attention
            i_idir_self_atten_embed = tf.concat([items_embeddings, i_idir_embeddings], 1)  # item_num,2dim

            i_idir_self_atten_embed_trans = tf.matmul(i_idir_self_atten_embed,
                                                     self.weights['item_node_attention_trans_w'])
            i_idir_cos_simi = tf.nn.leaky_relu(tf.squeeze(tf.matmul(i_idir_self_atten_embed_trans,
                                                                   tf.transpose(
                                                                       self.weights['item_node_attention_vector'],
                                                                       perm=[1, 0]))))  # item_num,

            item_node_attenton_score_ls += [i_idir_cos_simi]
            item_type_agg_all_embeddings += [i_idir_embeddings]

            # director idir attention
            d_idir_self_atten_embed = tf.concat([director_embeddings, d_idir_embeddings], 1)  # dir_num,2dim

            d_idir_self_atten_embed_trans = tf.matmul(d_idir_self_atten_embed,
                                                      self.weights['dir_node_attention_trans_w'])
            d_idir_cos_simi = tf.nn.leaky_relu(tf.squeeze(tf.matmul(d_idir_self_atten_embed_trans,
                                                                    tf.transpose(
                                                                        self.weights['dir_node_attention_vector'],
                                                                        perm=[1, 0]))))  # dir_num,

            dir_node_attenton_score_ls += [d_idir_cos_simi]
            dir_type_agg_all_embeddings += [d_idir_embeddings]

            i_id_A_fold_hat = self._split_A_hat_general(self.i_idir_pre_adj, self.n_items)
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(i_id_A_fold_hat[f], iidir_layer_embeddings))
            i_id_side_embeddings = tf.concat(temp_embed, 0)
            i_i_idir_embeddings = i_id_side_embeddings

            # item i_i ica attention
            i_i_idir_self_atten_embed = tf.concat([items_embeddings, i_i_idir_embeddings], 1)  # item_num,2dim

            i_i_idir_self_atten_embed_trans = tf.matmul(i_i_idir_self_atten_embed,
                                                       self.weights['item_node_attention_trans_w'])

            i_i_idir_cos_simi = tf.nn.leaky_relu(tf.squeeze(tf.matmul(i_i_idir_self_atten_embed_trans,
                                                                     tf.transpose(
                                                                         self.weights['item_node_attention_vector'],
                                                                         perm=[1, 0]))))  # item_num,

            item_node_attenton_score_ls += [i_i_idir_cos_simi]
            item_type_agg_all_embeddings += [i_i_idir_embeddings]

            # attention agg
            item_type_agg_all_embeddings = tf.stack(item_type_agg_all_embeddings, 0)  # 2,n_item,dim
            dir_type_agg_all_embeddings = tf.stack(dir_type_agg_all_embeddings, 0)  # 2,n_dir,dim

            item_node_attenton_score = tf.stack(item_node_attenton_score_ls, 0)  # 2,n_item
            dir_node_attenton_score = tf.stack(dir_node_attenton_score_ls, 0)  # 2,n_dir

            item_node_attenton_score = tf.nn.softmax(item_node_attenton_score, axis=0)  # 2,n_item

            dir_node_attenton_score = tf.nn.softmax(dir_node_attenton_score, axis=0)  # 2,n_dir

            item_node_attenton_score = tf.tile(tf.expand_dims(item_node_attenton_score, axis=2),
                                               multiples=[1, 1, self.emb_dim])  # 2,n_item,dim
            dir_node_attenton_score = tf.tile(tf.expand_dims(dir_node_attenton_score, axis=2),
                                             multiples=[1, 1, self.emb_dim])  # 2,n_dir,dim

            item_type_agg_embeddings = tf.reduce_sum(
                tf.multiply(item_type_agg_all_embeddings, item_node_attenton_score),
                axis=0, keepdims=False)  # n_item,dim
            dir_type_agg_embeddings = tf.reduce_sum(tf.multiply(dir_type_agg_all_embeddings, dir_node_attenton_score),
                                                   axis=0, keepdims=False)  # n_dir,dim

            users_dir_all_embeddings += [u_udir_embeddings]
            items_dir_all_embeddings += [item_type_agg_embeddings]

            users_embeddings = u_udir_embeddings
            items_embeddings = item_type_agg_embeddings
            director_embeddings = dir_type_agg_embeddings

            udir_layer_embeddings = tf.concat([users_embeddings, director_embeddings], axis=0)
            idir_layer_embeddings = tf.concat([items_embeddings, director_embeddings], axis=0)
            iidir_layer_embeddings = items_embeddings

        # u_i_dir attention
        users_dir_all_embeddings = tf.stack(users_dir_all_embeddings, axis=0)
        users_dir_final_embeddings = tf.reduce_mean(users_dir_all_embeddings, axis=0, keepdims=False)  # n_user,dim
        user_attention_vector = self.weights['user_attention_vector']  # n_user,dim
        # users_ca_final_embeddings_trans = tf.matmul(users_ca_final_embeddings, self.weights['uca_attention_trans_w'])
        users_dir_final_embeddings_trans = tf.matmul(users_dir_final_embeddings,
                                                    self.weights['ud_attention_trans_w']) + self.weights[
                                              'ud_attention_trans_b']
        users_dir_final_embeddings_trans = tf.nn.tanh(users_dir_final_embeddings_trans)
        user_dir_cos_simi = self.cosine_distance(users_dir_final_embeddings_trans, user_attention_vector)  # n_user,
        user_cos_score_ls += [user_dir_cos_simi]

        users_all_embeddings += [users_dir_final_embeddings]

        # i_dir attention
        items_dir_all_embeddings = tf.stack(items_dir_all_embeddings, axis=0)
        items_dir_final_embeddings = tf.reduce_mean(items_dir_all_embeddings, axis=0, keepdims=False)  # n_item,dim
        item_attention_vector = self.weights['item_attention_vector']  # n_item,dim
        # items_ca_final_embeddings_trans = tf.matmul(items_ca_final_embeddings, self.weights['ica_attention_trans_w'])
        items_dir_final_embeddings_trans = tf.matmul(items_dir_final_embeddings,
                                                     self.weights['id_attention_trans_w']) + self.weights[
                                               'id_attention_trans_b']
        items_dir_final_embeddings_trans = tf.nn.tanh(items_dir_final_embeddings_trans)

        item_dir_cos_simi = self.cosine_distance(items_dir_final_embeddings_trans, item_attention_vector)  # n_item,
        item_cos_score_ls += [item_dir_cos_simi]

        items_all_embeddings += [items_dir_final_embeddings]

        # u_i_type aggregate
        users_embeddings = self.weights['user_embedding']
        items_embeddings = self.weights['item_embedding']
        type_embeddings = self.weights['type_embedding']

        users_type_all_embeddings = [self.weights['user_embedding']]
        items_type_all_embeddings = [self.weights['item_embedding']]

        utype_layer_embeddings = tf.concat([users_embeddings, type_embeddings], axis=0)
        itype_layer_embeddings = tf.concat([items_embeddings, type_embeddings], axis=0)
        iitype_layer_embeddings = items_embeddings

        for k in range(0, self.n_layers):

            item_node_attenton_score_ls = []
            type_node_attenton_score_ls = []
            item_type_agg_all_embeddings = []
            type_type_agg_all_embeddings = []

            u_type_A_fold_hat = self._split_A_hat_general(self.u_type_pre_adj, self.n_users + self.n_type)
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(u_type_A_fold_hat[f], utype_layer_embeddings))

            utype_side_embeddings = tf.concat(temp_embed, 0)

            u_utype_embeddings, type_utype_embeddings = tf.split(utype_side_embeddings, [self.n_users, self.n_type], 0)

            # user utype attention

            # type utype attention
            type_utype_self_atten_embed = tf.concat([type_embeddings, type_utype_embeddings], 1)  # type_num,2dim

            type_utype_self_atten_embed_trans = tf.matmul(type_utype_self_atten_embed,
                                                      self.weights['type_node_attention_trans_w'])
            type_utype_cos_simi = tf.nn.leaky_relu(tf.squeeze(tf.matmul(type_utype_self_atten_embed_trans,
                                                                    tf.transpose(
                                                                        self.weights['type_node_attention_vector'],
                                                                        perm=[1, 0]))))  # type_num,

            type_node_attenton_score_ls += [type_utype_cos_simi]
            type_type_agg_all_embeddings += [type_utype_embeddings]

            i_t_A_fold_hat = self._split_A_hat_general(self.i_type_pre_adj, self.n_items + self.n_type)
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(i_t_A_fold_hat[f], itype_layer_embeddings))

            itype_side_embeddings = tf.concat(temp_embed, 0)

            i_itype_embeddings, type_itype_embeddings = tf.split(itype_side_embeddings, [self.n_items, self.n_type], 0)

            # item itype attention
            i_itype_self_atten_embed = tf.concat([items_embeddings, i_itype_embeddings], 1)  # item_num,2dim

            i_itype_self_atten_embed_trans = tf.matmul(i_itype_self_atten_embed,
                                                     self.weights['item_node_attention_trans_w'])
            i_itype_cos_simi = tf.nn.leaky_relu(tf.squeeze(tf.matmul(i_itype_self_atten_embed_trans,
                                                                   tf.transpose(
                                                                       self.weights['item_node_attention_vector'],
                                                                       perm=[1, 0]))))  # item_num,

            item_node_attenton_score_ls += [i_itype_cos_simi]
            item_type_agg_all_embeddings += [i_itype_embeddings]

            # type itype attention
            type_itype_self_atten_embed = tf.concat([type_embeddings, type_itype_embeddings], 1)  # type_num,2dim

            type_itype_self_atten_embed_trans = tf.matmul(type_itype_self_atten_embed,
                                                      self.weights['type_node_attention_trans_w'])
            type_itype_cos_simi = tf.nn.leaky_relu(tf.squeeze(tf.matmul(type_itype_self_atten_embed_trans,
                                                                    tf.transpose(
                                                                        self.weights['type_node_attention_vector'],
                                                                        perm=[1, 0]))))  # type_num,

            type_node_attenton_score_ls += [type_itype_cos_simi]
            type_type_agg_all_embeddings += [type_itype_embeddings]

            i_it_A_fold_hat = self._split_A_hat_general(self.i_itype_pre_adj, self.n_items)
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(i_it_A_fold_hat[f], iitype_layer_embeddings))
            i_it_side_embeddings = tf.concat(temp_embed, 0)
            i_i_itype_embeddings = i_it_side_embeddings

            # item i_i iba attention
            i_i_itype_self_atten_embed = tf.concat([items_embeddings, i_i_itype_embeddings], 1)  # item_num,2dim

            i_i_itype_self_atten_embed_trans = tf.matmul(i_i_itype_self_atten_embed,
                                                       self.weights['item_node_attention_trans_w'])
            i_i_itype_cos_simi = tf.nn.leaky_relu(tf.squeeze(tf.matmul(i_i_itype_self_atten_embed_trans,
                                                                     tf.transpose(
                                                                         self.weights['item_node_attention_vector'],
                                                                         perm=[1, 0]))))  # item_num,

            item_node_attenton_score_ls += [i_i_itype_cos_simi]
            item_type_agg_all_embeddings += [i_i_itype_embeddings]

            # attention agg
            item_type_agg_all_embeddings = tf.stack(item_type_agg_all_embeddings, 0)  # 2,n_item,dim
            type_type_agg_all_embeddings = tf.stack(type_type_agg_all_embeddings, 0)  # 2,n_type,dim

            item_node_attenton_score = tf.stack(item_node_attenton_score_ls, 0)  # 2,n_item
            type_node_attenton_score = tf.stack(type_node_attenton_score_ls, 0)  # 2,n_type

            item_node_attenton_score = tf.nn.softmax(item_node_attenton_score, axis=0)  # 2,n_item
            type_node_attenton_score = tf.nn.softmax(type_node_attenton_score, axis=0)  # 2,n_type

            item_node_attenton_score = tf.tile(tf.expand_dims(item_node_attenton_score, axis=2),
                                               multiples=[1, 1, self.emb_dim])  # 2,n_item,dim
            type_node_attenton_score = tf.tile(tf.expand_dims(type_node_attenton_score, axis=2),
                                             multiples=[1, 1, self.emb_dim])  # 2,n_type,dim

            item_type_agg_embeddings = tf.reduce_sum(
                tf.multiply(item_type_agg_all_embeddings, item_node_attenton_score),
                axis=0, keepdims=False)  # n_item,dim
            type_type_agg_embeddings = tf.reduce_sum(tf.multiply(type_type_agg_all_embeddings, type_node_attenton_score),
                                                   axis=0, keepdims=False)  # n_type,dim

            users_type_all_embeddings += [u_utype_embeddings]
            items_type_all_embeddings += [item_type_agg_embeddings]

            users_embeddings = u_utype_embeddings
            items_embeddings = item_type_agg_embeddings
            type_embeddings = type_type_agg_embeddings

            utype_layer_embeddings = tf.concat([users_embeddings, type_embeddings], axis=0)
            itype_layer_embeddings = tf.concat([items_embeddings, type_embeddings], axis=0)
            iitype_layer_embeddings = items_embeddings

        # u_type attention
        users_type_all_embeddings = tf.stack(users_type_all_embeddings, axis=0)
        users_type_final_embeddings = tf.reduce_mean(users_type_all_embeddings, axis=0, keepdims=False)  # n_user,dim
        user_attention_vector = self.weights['user_attention_vector']  # n_user,dim
        users_type_final_embeddings_trans = tf.matmul(users_type_final_embeddings,
                                                    self.weights['ut_attention_trans_w']) + self.weights[
                                              'ut_attention_trans_b']
        users_type_final_embeddings_trans = tf.nn.tanh(users_type_final_embeddings_trans)
        user_type_cos_simi = self.cosine_distance(users_type_final_embeddings_trans, user_attention_vector)  # n_user,
        user_cos_score_ls += [user_type_cos_simi]

        users_all_embeddings += [users_type_final_embeddings]

        # i_t attention
        items_t_all_embeddings = tf.stack(items_type_all_embeddings, axis=0)
        items_t_final_embeddings = tf.reduce_mean(items_t_all_embeddings, axis=0, keepdims=False)  # n_item,dim
        item_attention_vector = self.weights['item_attention_vector']  # n_item,dim
        items_t_final_embeddings_trans = tf.matmul(items_t_final_embeddings,
                                                     self.weights['it_attention_trans_w']) + self.weights[
                                               'it_attention_trans_b']
        items_t_final_embeddings_trans = tf.nn.tanh(items_t_final_embeddings_trans)

        item_t_cos_simi = self.cosine_distance(items_t_final_embeddings_trans, item_attention_vector)  # n_item,
        item_cos_score_ls += [item_t_cos_simi]

        items_all_embeddings += [items_t_final_embeddings]

        # attention
        users_all_embeddings = tf.stack(users_all_embeddings, 0)  # 4,n_user,dim
        items_all_embeddings = tf.stack(items_all_embeddings, 0)  # 4,n_item,dim

        user_cos_score = tf.stack(user_cos_score_ls, 0)  # 4,n_user
        item_cos_score = tf.stack(item_cos_score_ls, 0)  # 4,n_item

        user_atten_score = tf.nn.softmax(user_cos_score, axis=0)  # 4,n_user
        item_atten_score = tf.nn.softmax(item_cos_score, axis=0)  # 4,n_item

        user_atten_score = tf.tile(tf.expand_dims(user_atten_score, axis=2),
                                   multiples=[1, 1, self.emb_dim])  # 4,n_user,dim
        item_atten_score = tf.tile(tf.expand_dims(item_atten_score, axis=2),
                                   multiples=[1, 1, self.emb_dim])  # 4,n_item,dim

        users_final_embeddings = tf.reduce_sum(tf.multiply(users_all_embeddings, user_atten_score), axis=0,
                                               keepdims=False)  # n_user,dim
        items_final_embeddings = tf.reduce_sum(tf.multiply(items_all_embeddings, item_atten_score), axis=0,
                                               keepdims=False)  # n_item,dim

        return users_final_embeddings, items_final_embeddings

    def cosine_distance(self, x1, x2):
        x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1))
        x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1))

        x1_x2 = tf.reduce_sum(tf.multiply(x1, x2), axis=1)
        cosin = x1_x2 / (x1_norm * x2_norm)
        return cosin

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
            self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre) +  \
                      tf.nn.l2_loss(self.weights['actor_embedding']) + \
                      tf.nn.l2_loss(self.weights['director_embedding']) + \
                      tf.nn.l2_loss(self.weights['type_embedding'])


        regularizer = regularizer / self.batch_size

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)


def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data


# parallelized sampling on CPU
class sample_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        with tf.device(cpus[0]):
            self.data = data_generator.sample()


class sample_thread_test(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        with tf.device(cpus[0]):
            self.data = data_generator.sample_test()


# training on GPU
class train_thread(threading.Thread):
    def __init__(self, model, sess, sample):
        threading.Thread.__init__(self)
        self.model = model
        self.sess = sess
        self.sample = sample
    def run(self):
        if len(gpus):
            with tf.device(gpus[-1]):
                users, pos_items, neg_items = self.sample.data
                self.data = sess.run(
                    [self.model.opt, self.model.loss, self.model.mf_loss, self.model.emb_loss, self.model.reg_loss],
                    feed_dict={model.users: users, model.pos_items: pos_items,
                               model.node_dropout: eval(args.node_dropout),
                               model.mess_dropout: eval(args.mess_dropout),
                               model.neg_items: neg_items})
        else:
            users, pos_items, neg_items = self.sample.data
            self.data = sess.run(
                [self.model.opt, self.model.loss, self.model.mf_loss, self.model.emb_loss, self.model.reg_loss],
                feed_dict={model.users: users, model.pos_items: pos_items,
                           model.node_dropout: eval(args.node_dropout),
                           model.mess_dropout: eval(args.mess_dropout),
                           model.neg_items: neg_items})


class train_thread_test(threading.Thread):
    def __init__(self, model, sess, sample):
        threading.Thread.__init__(self)
        self.model = model
        self.sess = sess
        self.sample = sample

    def run(self):
        if len(gpus):
            with tf.device(gpus[-1]):
                users, pos_items, neg_items = self.sample.data
                self.data = sess.run([self.model.loss, self.model.mf_loss, self.model.emb_loss],
                                     feed_dict={model.users: users, model.pos_items: pos_items,
                                                model.neg_items: neg_items,
                                                model.node_dropout: eval(args.node_dropout),
                                                model.mess_dropout: eval(args.mess_dropout)})
        else:
            users, pos_items, neg_items = self.sample.data
            self.data = sess.run([self.model.loss, self.model.mf_loss, self.model.emb_loss],
                                 feed_dict={model.users: users, model.pos_items: pos_items,
                                            model.neg_items: neg_items,
                                            model.node_dropout: eval(args.node_dropout),
                                            model.mess_dropout: eval(args.mess_dropout)})


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    f0 = time()

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_actor'] = data_generator.n_actor
    config['n_director'] = data_generator.n_director
    config['n_type'] = data_generator.n_type

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, ui_pre_adj = data_generator.get_adj_mat()

    u_actor_adj_mat, u_actor_norm_adj_mat, u_actor_mean_adj_mat, u_actor_pre_adj_mat = data_generator.\
        create_graph_adj_mat(data_generator.n_users, data_generator.n_actor, data_generator.R_u_actor)

    u_di_adj_mat, u_di_norm_adj_mat, u_di_mean_adj_mat, u_di_pre_adj_mat = data_generator.create_graph_adj_mat(
        data_generator.n_users, data_generator.n_director, data_generator.R_u_di)

    u_type_adj_mat, u_type_norm_adj_mat, u_type_mean_adj_mat, u_type_pre_adj_mat = data_generator.create_graph_adj_mat(
        data_generator.n_users, data_generator.n_type, data_generator.R_u_type)

    i_actor_adj_mat, i_actor_norm_adj_mat, i_actor_mean_adj_mat, i_actor_pre_adj_mat = data_generator.\
        create_graph_adj_mat(data_generator.n_items, data_generator.n_actor, data_generator.R_i_actor)

    i_di_adj_mat, i_di_norm_adj_mat, i_di_mean_adj_mat, i_di_pre_adj_mat = data_generator.create_graph_adj_mat(
        data_generator.n_items, data_generator.n_director, data_generator.R_i_dir)

    i_type_adj_mat, i_type_norm_adj_mat, i_type_mean_adj_mat, i_type_pre_adj_mat = data_generator.create_graph_adj_mat(
        data_generator.n_items, data_generator.n_type, data_generator.R_i_type)

    i_ia_adj_mat, i_ia_norm_adj_mat, i_ia_mean_adj_mat, i_ia_pre_adj_mat = data_generator.create_graph_adj_mat(
        data_generator.n_items, data_generator.n_items, data_generator.R_i_iactor, flag=1)

    i_idir_adj_mat, i_idir_norm_adj_mat, i_idir_mean_adj_mat, i_idir_pre_adj_mat = data_generator.create_graph_adj_mat(
        data_generator.n_items, data_generator.n_items, data_generator.R_i_idir, flag=1)

    i_itype_adj_mat, i_itype_norm_adj_mat, i_itype_mean_adj_mat, i_itype_pre_adj_mat = data_generator.create_graph_adj_mat(
        data_generator.n_items, data_generator.n_items, data_generator.R_i_itype, flag=1)

    config['ui_norm_adj'] = ui_pre_adj
    config['u_actor_pre_adj_mat'] = u_actor_pre_adj_mat
    config['u_di_pre_adj_mat'] = u_di_pre_adj_mat
    config['u_type_pre_adj_mat'] = u_type_pre_adj_mat
    config['i_actor_pre_adj_mat'] = i_actor_pre_adj_mat
    config['i_di_pre_adj_mat'] = i_di_pre_adj_mat
    config['i_type_pre_adj_mat'] = i_type_pre_adj_mat
    config['i_ia_pre_adj_mat'] = i_ia_pre_adj_mat
    config['i_idir_pre_adj_mat'] = i_idir_pre_adj_mat
    config['i_itype_pre_adj_mat'] = i_itype_pre_adj_mat

    print('use the pre adjcency matrix')

    t0 = time()
    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None
    model = CMoAN_movie(data_config=config, pretrain_data=pretrain_data)

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])

        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            if args.report != 1:
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=True)
                cur_best_pre_0 = ret['recall'][0]

                pretrain_ret = 'pretrained model recall=[%s], precision=[%s], ' \
                               'ndcg=[%s]' % \
                               (', '.join(['%.5f' % r for r in ret['recall']]),
                                ', '.join(['%.5f' % r for r in ret['precision']]),
                                ', '.join(['%.5f' % r for r in ret['ndcg']]))
                print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    *********************************************************
    Get the performance w.r.t. different sparsity levels.
    """
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split()
        users_to_test_list.append(list(data_generator.test_set.keys()))
        split_state.append('all')

        report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(report_path)
        f = open(report_path, 'w')
        f.write(
            'embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, args.adj_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test, drop_flag=True)

            final_perf = "recall=[%s], precision=[%s], ndcg=[%s]" % \
                         (', '.join(['%.5f' % r for r in ret['recall']]),
                          ', '.join(['%.5f' % r for r in ret['precision']]),
                          ', '.join(['%.5f' % r for r in ret['ndcg']]))

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    Train.
    """
    tensorboard_model_path = 'tensorboard/'
    if not os.path.exists(tensorboard_model_path):
        os.makedirs(tensorboard_model_path)
    run_time = 1
    while (True):
        if os.path.exists(tensorboard_model_path + model.log_dir + '/run_' + str(run_time)):
            run_time += 1
        else:
            break
    train_writer = tf.summary.FileWriter(tensorboard_model_path + model.log_dir + '/run_' + str(run_time), sess.graph)

    loss_loger, map_loger, ndcg_loger, hit_loger,recall_loger,precision_loger =  [], [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(1, args.epoch + 1):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        train_n_batch = data_generator.n_train // args.batch_size + 1
        test_n_batch = data_generator.n_test // args.batch_size + 1
        loss_test, mf_loss_test, emb_loss_test, reg_loss_test = 0., 0., 0., 0.

        '''
        *********************************************************
        parallelized sampling
        '''
        train_users, train_pos_items, train_neg_items = data_generator.get_train_data_pairs(neg_sample_num=args.neg_sample)
        train_data_nums = len(train_neg_items)

        for idx in range(train_n_batch):
            start = args.batch_size * idx
            end = min(args.batch_size * (idx + 1), train_data_nums)
            batch_users = train_users[start:end]
            batch_pos_items = train_pos_items[start:end]
            batch_neg_items = train_neg_items[start:end]

            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run(
                [model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
                feed_dict={model.users: batch_users, model.pos_items: batch_pos_items,
                           model.node_dropout: eval(args.node_dropout),
                           model.mess_dropout: eval(args.mess_dropout),
                           model.neg_items: batch_neg_items})

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch % 10) != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue
        users_to_test = list(data_generator.train_items.keys())
        ret = test2(sess, model, users_to_test, drop_flag=True, train_set_flag=1)

        perf_str = 'Epoch %d: train==[%.5f=%.5f + %.5f + %.5f], map=[%s], ndcg=[%s], hit_ratio=[%s],recall=[%s], precision=[%s]' % \
                   (epoch, loss, mf_loss, emb_loss, reg_loss,
                    ', '.join(['%.5f' % ret['map']]),
                    ', '.join(['%.5f' % ret['ndcg']]),
                    ', '.join(['%.5f' % ret['hit_ratio']]),
                    ', '.join(['%.5f' % ret['recall']]),
                    ', '.join(['%.5f' % ret['precision']]))
        print(perf_str)

        '''
        *********************************************************
        parallelized sampling
        '''
        test_users, test_pos_items, test_neg_items = data_generator.get_test_data_pairs()
        test_data_nums = len(test_neg_items)

        for idx in range(test_n_batch):
            start = args.batch_size * idx
            end = min(args.batch_size * (idx + 1), test_data_nums)
            batch_users = test_users[start:end]
            batch_pos_items = test_users[start:end]
            batch_neg_items = test_users[start:end]

            batch_loss_test, batch_mf_loss_test, batch_emb_loss_test = sess.run([model.loss, model.mf_loss, model.emb_loss],
                                            feed_dict={model.users: batch_users, model.pos_items: batch_pos_items,
                                            model.neg_items: batch_neg_items,
                                            model.node_dropout: eval(args.node_dropout),
                                            model.mess_dropout: eval(args.mess_dropout)})

            loss_test += batch_loss_test
            mf_loss_test += batch_mf_loss_test
            emb_loss_test += batch_emb_loss_test

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test2(sess, model, users_to_test, drop_flag=True)

        t3 = time()

        loss_loger.append(loss)
        map_loger.append(ret['map'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])
        recall_loger.append(ret['recall'])
        precision_loger.append(ret['precision'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f + %.5f], map=[%s], ' \
                       'ndcg=[%s], hit_ratio=[%s],recall=[%s], precision=[%s]' % \
                       (epoch, t2 - t1, t3 - t2, loss_test, mf_loss_test, emb_loss_test, reg_loss_test,
                        ', '.join(['%.5f' % r for r in ret['map']]),
                        ', '.join(['%.5f' % r for r in ret['ndcg']]),
                        ', '.join(['%.5f' % r for r in ret['hit_ratio']]),
                        ', '.join(['%.5f' % r for r in ret['recall']]),
                        ', '.join(['%.5f' % r for r in ret['precision']]))
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    maps = np.array(map_loger)
    ndcgs = np.array(ndcg_loger)
    hit_ratings = np.array(hit_loger)
    recalls = np.array(recall_loger)
    precisions = np.array(precision_loger)

    best_rec_0 = max(recalls[:, 0])
    idx = list(recalls[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\tmap=[%s], ndcg=[%s], hit_ratio=[%s],recall=[%s], precision=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in maps[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]),
                  '\t'.join(['%.5f' % r for r in hit_ratings[idx]]),
                  '\t'.join(['%.5f' % r for r in recalls[idx]]),
                  '\t'.join(['%.5f' % r for r in precisions[idx]]))

    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()
