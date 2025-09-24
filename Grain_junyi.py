import scipy.sparse as sp
import numpy as np
import torch
import params

dataset = 'data/junyi/'
with open(dataset + 'grain.txt') as f:
    f.readline()
    tn, an = f.readline().split(',')
    tn, an = int(tn), int(an)
with open(dataset+'name_topic.txt', 'r') as f:  # name topic
    dict_nt = {}
    for line in f.readlines():
        line = line.replace('\n', '').split('\t')
        dict_nt[int(line[0])] = int(line[1])
with open(dataset+'topic_area.txt', 'r') as f:  # topic area
    dict_ta = {}
    for line in f.readlines():
        line = line.replace('\n', '').split('\t')
        dict_ta[int(line[0])] = int(line[1])


def equal(list_kn):
    output_kn = []
    for i in list_kn:
        output_kn.append(i)
        if int(i) in dict_nt:
            topic = dict_nt[int(i)]
            output_kn.append(topic)
            if int(topic) in dict_ta:
                area = dict_ta[int(topic)]
                output_kn.append(area)
    return output_kn


# def top(list_kn):
#     output_kn = []
#     for i in list_kn:
#         if int(i) in dict_nt:
#             topic = dict_nt[int(i)]
#             if int(topic) in dict_ta:
#                 area = dict_ta[int(topic)]
#                 output_kn.append(area)
#     return output_kn


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adjM(txt):
    edges = np.genfromtxt(txt,dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(params.kn, params.kn),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def mA(txt):
    edges = np.genfromtxt(txt,dtype=np.int32)
    if txt==dataset+'topic_area.txt':
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 1]-835-params.tn, edges[:, 0]-835)),
                            shape=(params.an, params.tn),
                            dtype=np.float32)
    elif txt==dataset+'name_topic.txt':
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 1]-835, edges[:, 0])),
                            shape=(params.tn, 835),
                            dtype=np.float32)
    else:
        raise ValueError("无此层间关系")
    # build symmetric adjacency matrix
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj.to_dense()

mAT = mA(dataset+'topic_area.txt')
mTN = mA(dataset+'name_topic.txt')


if params.type == "equal":
    adjNT = adjM(dataset+'name_topic.txt')
    adjTA = adjM(dataset+'topic_area.txt')
    BERT_emb = np.load('data/junyi/BERT_emb1.npy', allow_pickle=True)

