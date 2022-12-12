import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from scipy import sparse

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data_2(features, head,max_len):
    # f = open('./123.txt', 'w')
    # f.write(edge_str)
    # f.close()
    # f2 = open('./123.txt', 'r')

    # f2=edge_str
    # 边
    # edges_unordered = np.genfromtxt(f2, dtype=np.int32)
    #
    # edges_unordered_2 = edges_unordered.flatten()
    # # 边
    # edges = np.array(edges_unordered_2,
    #                  dtype=np.int32).reshape(edges_unordered.shape)  # get 去键值，flatten 展评操作（一维）,map把后边的值用来搜索键值
    # lines = f2.readlines()
    # if lines is None:
    #     edges = np.empty(shape=[2, 2])
    #     a=[1,1]
    #     a=np.array(a)
    #     edges[0] = a
    #     edges[1] = a
    # else:
    #     # 如果是一维数组
    #     if edges.ndim==1:
    #         a=[1,1]
    #         a = np.array(a)
    #         # 创建一维数组
    #         edges=np.empty(shape=[2,2])
    #         edges[0]=a
    #         edges[1]=a
    # 邻接矩阵
    head_list=head
    # 从head_list到邻接矩阵
    def head_to_adj(head, max_sent_len):
        ret = np.zeros((max_sent_len, max_sent_len), dtype=np.float32)
        for i in range(len(head)):
            j = head[i]
            if j != 0:
                ret[i, j - 1] = 1
                ret[j - 1, i] = 1
        return ret
    #adj
    adj=head_to_adj(head_list, max_len) # ndarray

    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(18, 18),
    #                     dtype=np.float32)  # 在(edges[:, 0], edges[:, 1])对应行列上生成(np.ones(edges.shape[0]) 1
    adj = sparse.csr_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    #feature

    # f4 = open('./zz2.tsv', 'r')
    # idx_features_labels = np.genfromtxt(f4, dtype=np.dtype(str))  # 变成ndarray数组
    #features = sp.csr_matrix(idx_features_labels, dtype=np.float32)
    # features = sp.csr_matrix(features, dtype=np.float32)
    # features = normalize(features)
    # features = torch.FloatTensor(np.array(features.todense()))
    # features = features.t()

    # build symmetric adjacency matrix
    #f2.close()
    idx_train = range(15)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # labels
    f5 = open('./label.txt', 'r')
    labels = np.genfromtxt(f5, dtype=np.int32)
    labels = torch.Tensor(labels)
    f5.close()

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # adj=torch.LongTensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    #features=torch.LongTensor(features)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    f = open('./tu_dict.tsv', 'r')  # 网络

    # protein_id_name = {}  # 字典存放蛋白质：key为蛋白质编号，value为蛋白质名称
    # protein_id = {}  # 字典存放蛋白质：key为蛋白质编号，value为蛋白质名称
    # protein_name_id = {}  # 与上相同，key和value相同
    # adjacent_list = defaultdict(set)  # 存放无权网络中的邻接矩阵
    # lines = f.readlines()  # 读取蛋白质网络
    # count = 0
    # for line in lines:  # 根据读取的蛋白质网络在字典中存放蛋白质
    #     words = line.strip().split('\t')
    #     for i in range(len(words)):
    #         if words[i] not in protein_name_id:
    #             protein_name_id[words[i]] = count
    #             protein_id_name[count] = words[i]
    #             count += 1
    # print('共有蛋白质对的个数为:' + str(count))
    # my_count = 0
    # for line in lines:  # 根据读取的蛋白质网络在字典中存放蛋白质
    #     words = line.strip().split('\t')
    #     for i in range(len(words)):
    #         if words[i] not in protein_id:
    #             protein_id[words[i]] = my_count  # 2178
    #             my_count = my_count + 1

    # 边
    f2 = open('./ccc.tsv', 'r')
    edges_unordered = np.genfromtxt(f2, dtype=np.int32)
    edges_unordered_2 = edges_unordered.flatten()
    # 边
    edges = np.array(edges_unordered_2,
                     dtype=np.int32).reshape(edges_unordered.shape)  # get 去键值，flatten 展评操作（一维）,map把后边的值用来搜索键值
    a = np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])

    # 邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(3100, 3100),
                        dtype=np.float32)  # 在(edges[:, 0], edges[:, 1])对应行列上生成(np.ones(edges.shape[0]) 1

    # feature
    f4 = open('./zz2.tsv', 'r')
    idx_features_labels = np.genfromtxt(f4, dtype=np.dtype(str))  # 变成ndarray数组
    features = sp.csr_matrix(idx_features_labels, dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))

    # labels
    f5 = open('./labels.tsv', 'r')
    labels = np.genfromtxt(f5, dtype=np.int32)
    labels = torch.Tensor(labels)
    f5.close()

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # adj=torch.LongTensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


# def load_data(path="../data/cora/", dataset="cora"):
#     """Load citation network dataset (cora only for now)"""
#     print('Loading {} dataset...'.format(dataset))
#
#     # a = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7]])
#     # b = a[:, 1:-1]
#     # c = sp.csr_matrix(b, dtype=np.float32)
#     #
#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
#                                         dtype=np.dtype(str))  # 变成ndarray数组
#     ididid=idx_features_labels[:, 1:-1]
#     # features = sp.csr_matrix(ididid, dtype=np.float32)
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     a=idx_features_labels[:, -1]
#     labels = encode_onehot(idx_features_labels[:, -1])
#
#     #features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     #labels = encode_onehot(idx_features_labels[:, -1])
#
#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)} #2078
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
#                                     dtype=np.int32)
#     a=list(map(idx_map.get, edges_unordered.flatten()))
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)  # get 去键值，flatten 展评操作（一维）,map把后边的值用来搜索键值
#
#     a=edges[:, 0]
#     a=np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])
#     a=labels.shape[0]
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]),
#                         dtype=np.float32)  # 在(edges[:, 0], edges[:, 1])对应行列上生成(np.ones(edges.shape[0]) 1
#
#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#
#     features = normalize(features)
#     adj = normalize(adj + sp.eye(adj.shape[0]))
#
#     idx_train = range(140)
#     idx_val = range(200, 500)
#     idx_test = range(500, 1500)
#
#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(np.where(labels)[1])
#     adj = sparse_mx_to_torch_sparse_tensor(adj)
#
#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)
#
#     return adj, features, labels, idx_train, idx_val, idx_test
#

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
