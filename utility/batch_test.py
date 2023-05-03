
from utility.parser import parse_args
from utility.load_data import *
from evaluator import eval_score_matrix_foldout
import multiprocessing
import heapq
import numpy as np
import metrics
cores = multiprocessing.cpu_count() // 2

args = parse_args()

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test

BATCH_SIZE = args.batch_size

def test2(sess, model, users_to_test, drop_flag=False, train_set_flag=0):
    # B: batch size
    # N: the number of items
    result = {'map': 0., 'ndcg': 0., 'hit_ratio': 0., 'recall': 0.,'precision': 0.}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    item_batch = range(ITEM_NUM)
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        if drop_flag == False:
            rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                        model.pos_items: item_batch})
        else:
            rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                        model.pos_items: item_batch,
                                                        model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                        model.mess_dropout: [0.] * len(eval(args.layer_size))})
        rate_batch = np.array(rate_batch)  # (B, N)

        user_batch_rating_uid = zip(rate_batch, user_batch)
        if train_set_flag == 0:
            batch_result = pool.map(test_one_user_test, user_batch_rating_uid)
        else:
            batch_result = pool.map(test_one_user_train, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['map'] += re['map']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall'] / n_test_users

    assert count == n_test_users
    pool.close()
    if train_set_flag == 0:
        result['hit_ratio'] = result['hit_ratio'] / N_TEST
    else:
        result['hit_ratio'] = result['hit_ratio'] / N_TRAIN

    return result


def test_one_user_train(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    user_pos_test = data_generator.train_items[u]
    test_items = list(range(ITEM_NUM))

    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, int(args.Ks))
    # else:
    #     r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, int(args.Ks))


def test_one_user_test(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, int(args.Ks))

    return get_performance(user_pos_test, r, auc, int(args.Ks))


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max_item_score = heapq.nlargest(Ks, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    map, ndcg, hit_ratio,precision, recall = [], [], [], [], []

    map.append(metrics.average_precision(r, Ks))
    ndcg.append(metrics.ndcg_at_k(r, Ks))
    hit_ratio.append(metrics.hit_at_k(r, Ks))
    precision.append(metrics.precision_at_k(r, Ks))
    recall.append(metrics.recall_at_k(r, Ks, len(user_pos_test)))

    return {'map': np.array(map),'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio),
            'recall': np.array(recall),'precision': np.array(precision)}








