from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy, load_data_2
from pygcn.models import GCN
import torch.nn as nn

# Training settings
def my_init():
    parser2 = argparse.ArgumentParser()
    parser2.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser2.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser2.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser2.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser2.add_argument('--lr', type=float, default=2e-5,
                        help='Initial learning rate.')
    parser2.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser2.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser2.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    args2 = parser2.parse_args()
    args2.cuda = not args2.no_cuda and torch.cuda.is_available()

    np.random.seed(args2.seed)
    torch.manual_seed(args2.seed)
    if args2.cuda:
        torch.cuda.manual_seed(args2.seed)
    return args2

#
def dada(args2,features,head_list,max_len):
    # Load data
    adj,features,labels,idx_train, idx_val, idx_test = load_data_2(features,head_list,max_len)
    if args2.cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    #Model and optimizer
    # model_g = GCN(nfeat=features.shape[1],
    #             nhid=args2.hidden,
    #             nclass=5,
    #             dropout=args2.dropout)
    #
    # optimizer2 = optim.Adam(model_g.parameters(),
    #                        lr=args2.lr, weight_decay=args2.weight_decay)
    return adj, features, labels, idx_train, idx_val, idx_test


# def train(self,args2, adj, features, labels, idx_train, idx_val, idx_test,label):
#     if args2.cuda:
#         #model_g.cuda()
#         features = features.cuda()
#         adj = adj.cuda()
#         labels = labels.cuda()
#         idx_train = idx_train.cuda()
#         idx_val = idx_val.cuda()
#         idx_test = idx_test.cuda()
#     # model_g.zero_grad()
#     # model_g.train()
#     # #optimizer2.zero_grad()
#     # output_g = model_g(features, adj)#400*5->1*5
#     # m22 = nn.LayerNorm(output_g.size()[0:]).cuda()
#     # output_g = m22(output_g)
#     # optimizer2.step()
#     #output = torch.mean(output, 1)
#     #loss_fct = nn.CrossEntropyLoss()
#     #loss_train = loss_fct(output.view(-1, 5), label.view(-1))
#     #loss_train = F.nll_loss(output, labels)
#     #acc_train = accuracy(output, labels)
#     #loss_train.backward()
#     #optimizer.step()
#     # t = time.time()
#     # for i in range(9):
#     #     model.train()
#     #     optimizer.zero_grad()
#     #     output = model(features, adj)
#     #     output = torch.mean(output, 0)
#     #
#     #     # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
#     #     # if labels is not None:
#     #     #     if self.num_labels == 1:
#     #     #         loss_fct = nn.MSELoss()
#     #     #         loss = loss_fct(logits.view(-1), labels.view(-1))
#     #     #     else:
#     #     #         loss_fct = nn.CrossEntropyLoss()
#     #     #
#     #     #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#     #     #
#     #     #
#     #     #     outputs = (loss,) + outputs
#     #     loss_fct = nn.CrossEntropyLoss()
#     #     loss_train = loss_fct(output.view(-1, 5), label.view(-1))
#     #     # loss_train = F.nll_loss(output, labels)
#     #     # acc_train = accuracy(output, labels)
#     #     loss_train.backward(retain_graph=True)
#     #     optimizer.step()
#
#     # 正常跑代码的时候加上
#     # loss_train = F.nll_loss(output[idx_train], labels[idx_train].long())
#     # acc_train = accuracy(output[idx_train], labels[idx_train])
#     # loss_train.backward()
#     # optimizer.step()
#
#     # if not args.fastmode:
#     #     # Evaluate validation set performance separately,
#     #     # deactivates dropout during validation run.
#     #     model.eval()
#     #     output = model(features, adj)
#     #
#     # loss_val = F.nll_loss(output[idx_val], labels[idx_val].long())
#     # acc_val = accuracy(output[idx_val], labels[idx_val])
#     # print('Epoch: {:04d}'.format(epoch + 1),
#     #       'loss_train: {:.4f}'.format(loss_train.item()),
#     #       'acc_train: {:.4f}'.format(acc_train.item()),
#     #       'loss_val: {:.4f}'.format(loss_val.item()),
#     #       'acc_val: {:.4f}'.format(acc_val.item()),
#     #       'time: {:.4f}s'.format(time.time() - t))
#     return output_g


# def train2(args, model_g, optimizer2, adj, features, labels, idx_train, idx_val, idx_test,label):
#     if args.cuda:
#         model_g.cuda()
#         features = features.cuda()
#         adj = adj.cuda()
#         labels = labels.cuda()
#         idx_train = idx_train.cuda()
#         idx_val = idx_val.cuda()
#         idx_test = idx_test.cuda()
#     model_g.eval()
#     output_g = model_g(features, adj)#400*5->1*5
#     m22 = nn.LayerNorm(output_g.size()[0:]).cuda()
#     output_g = m22(output_g)
#     #output = torch.mean(output, 1)
#     #loss_fct = nn.CrossEntropyLoss()
#     #loss_train = loss_fct(output.view(-1, 5), label.view(-1))
#     #loss_train = F.nll_loss(output, labels)
#     #acc_train = accuracy(output, labels)
#     #loss_train.backward()
#     #optimizer.step()
#     # t = time.time()
#     # for i in range(9):
#     #     model.train()
#     #     optimizer.zero_grad()
#     #     output = model(features, adj)
#     #     output = torch.mean(output, 0)
#     #
#     #     # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
#     #     # if labels is not None:
#     #     #     if self.num_labels == 1:
#     #     #         loss_fct = nn.MSELoss()
#     #     #         loss = loss_fct(logits.view(-1), labels.view(-1))
#     #     #     else:
#     #     #         loss_fct = nn.CrossEntropyLoss()
#     #     #
#     #     #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#     #     #
#     #     #
#     #     #     outputs = (loss,) + outputs
#     #     loss_fct = nn.CrossEntropyLoss()
#     #     loss_train = loss_fct(output.view(-1, 5), label.view(-1))
#     #     # loss_train = F.nll_loss(output, labels)
#     #     # acc_train = accuracy(output, labels)
#     #     loss_train.backward(retain_graph=True)
#     #     optimizer.step()
#
#     # 正常跑代码的时候加上
#     # loss_train = F.nll_loss(output[idx_train], labels[idx_train].long())
#     # acc_train = accuracy(output[idx_train], labels[idx_train])
#     # loss_train.backward()
#     # optimizer.step()
#
#     # if not args.fastmode:
#     #     # Evaluate validation set performance separately,
#     #     # deactivates dropout during validation run.
#     #     model.eval()
#     #     output = model(features, adj)
#     #
#     # loss_val = F.nll_loss(output[idx_val], labels[idx_val].long())
#     # acc_val = accuracy(output[idx_val], labels[idx_val])
#     # print('Epoch: {:04d}'.format(epoch + 1),
#     #       'loss_train: {:.4f}'.format(loss_train.item()),
#     #       'acc_train: {:.4f}'.format(acc_train.item()),
#     #       'loss_val: {:.4f}'.format(loss_val.item()),
#     #       'acc_val: {:.4f}'.format(acc_val.item()),
#     #       'time: {:.4f}s'.format(time.time() - t))
#     return output_g
# Train model
def my_train(args2, features,head_list,max_len):
    # t_total = time.time()
    adj, features,labels, idx_train, idx_val, idx_test= dada(args2, features,head_list,max_len)
    # for epoch in range(args.epochs):
    #output=train(epoch, args, model, optimizer, adj, features, labels, idx_train, idx_val, idx_test)
    #output_g=train(args2, adj, features, labels, idx_train, idx_val, idx_test,label)
    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    return adj,features

# def my_train2(args2, features,head,max_len,label):
#     # t_total = time.time()
#     model_g, optimizer2, adj, features,labels, idx_train, idx_val, idx_test,label = dada(args2, features,head,max_len,label)
#     # for epoch in range(args.epochs):
#     #output=train(epoch, args, model, optimizer, adj, features, labels, idx_train, idx_val, idx_test)
#     output_g=train2(args2, model_g, optimizer2, adj, features, labels, idx_train, idx_val, idx_test,label)
#     # print("Optimization Finished!")
#     # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
#     return output_g

def grap_h(feature,head,max_len):
    args2 = my_init()
    output_g=my_train(args2,feature,head,max_len,)
    return output_g

# def grap_h2(feature,head,max_len,labels):
#     args2 = my_init()
#     output_g=my_train2(args2,feature,head,max_len,labels)
#     return output_g



if __name__ == '__main__':
    c=[('ROOT', 0, 4), ('det', 3, 1), ('compound', 3, 2), ('nsubj', 4, 3), ('case', 6, 5), ('obl', 4, 6), (
    'punct', 4, 7), ('advmod', 10, 8), ('case', 10, 9), ('obl', 4, 10), ('punct', 4, 11), ('dep', 4, 12), (
          'mark', 14, 13), ('advcl', 12, 14), ('amod', 16, 15), ('obj', 14, 16), ('punct', 4, 17), ('advmod', 19, 18), (
          'parataxis', 4, 19), ('compound', 22, 20), ('amod', 22, 21), ('obj', 19, 22), ('punct', 4, 23)]
    a=[101, 138, 15385, 1538, 2881, 5128, 1106, 12890, 9871, 19840, 117, 1932, 1112, 19273, 132, 1215, 1106, 7299, 5250, 27372, 15242, 1643, 7580, 4206, 132, 1145, 1144, 8432, 9760, 4625, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    b=[3, 3, 4, 0, 6, 4, 4, 10, 10, 4, 4, 4, 14, 12, 16, 14, 4, 19, 4, 22, 22, 19, 4]
    grap_h(a,b,c)
