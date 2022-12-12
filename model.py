import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
import numpy as np
import pandas as pd
from pygcn import train
from transformers.modeling_bert import BertSelfAttention, BertLayer, BertAttention


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(RBERT, self).__init__(bert_config)
        self.bert = BertModel.from_pretrained(args.pretrained_model_name, config=bert_config)  # Load pretrained bert

        self.num_labels = bert_config.num_labels

        self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)  # fc全连接
        self.e1_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.e2_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        # self.label_classifier = FCLayer(bert_config.hidden_size * 3, bert_config.num_labels, args.dropout_rate,
        #                                 use_activation=False)
        # self.label_classifier2 = FCLayer(bert_config.hidden_size * 4, bert_config.num_labels, args.dropout_rate,
        #                                 use_activation=False)
        self.label_classifier = FCLayer(bert_config.hidden_size * 3, bert_config.num_labels, args.dropout_rate,
                                        )
        self.label_classifier2 = FCLayer(bert_config.hidden_size * 4, bert_config.num_labels, args.dropout_rate,
                                         use_activation=False)
        self.out_layer = FCLayer(bert_config.hidden_size * 3, bert_config.hidden_size * 3, args.dropout_rate)

        self.gcntensor = FCLayer(5, 768, args.dropout_rate)

        self.gcn_fc_layer = FCLayer(400, 768, args.dropout_rate)

        config_s = BertConfig()
        self.att1 = BertAttention(config_s)

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]   unsqueeze()--->扩充维度
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(
            1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    # 当调用trainer.py outputs = self.model(**inputs)执行
    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, li, li2, flag):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        li = li.tolist()
        li2 = li2.tolist()
        li22 = li2[0]
        ll = li[0] - 2
        ii = 0
        for i in range(ll, 400):
            if li22[i] == int(0):
                ii = i
                break
        del li22[ii:400]

        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # 新加
        e1_arr = []
        e2_arr = []
        # 计算e1_id, e2_id
        e2_info = []
        e1_info = []

        e1info_attention_mask = []
        e2info_attention_mask = []
        batch = e1_h.shape[0]
        e1_id = (input_ids * e1_mask).cpu().numpy().tolist()
        e2_id = (input_ids * e2_mask).cpu().numpy().tolist()

        e1_head_list = [[] for i in range(batch)]
        e1_max_len_list = [[] for i in range(batch)]
        e2_head_list = [[] for i in range(batch)]
        e2_max_len_list = [[] for i in range(batch)]

        # 额外信息
        f_extra = open('data/extra_dict.tsv', 'r')
        # f2 = open('data/allextra.tsv', 'r')
        lines = f_extra.readlines()
        # lines2 = f2.readlines()

        for i in range(batch):
            #  提取实体
            a = pd.DataFrame(e1_id[i]).replace(0, np.NAN)
            a.dropna(inplace=True)
            a = np.array(a).astype(np.int32).reshape(1, -1)[0]
            a = a.tolist()
            # 删除最后一个109
            a.pop()
            # 删除第一个109
            del (a[0])

            e1_arr.append(a)

            b = pd.DataFrame(e2_id[i]).replace(0, np.NAN)
            b.dropna(inplace=True)
            b = np.array(b).astype(np.int32).reshape(1, -1)[0]
            b = b.tolist()
            # 删除最后一个109
            b.pop()
            # 删除第一个109
            del (b[0])
            e2_arr.append(b)

        # 在extra文件中相同的实体，如果找到返回后边的实体解释

        for i in range(batch):
            empty_list = [1] * 400
            ini_attention_mask = [0] * 400
            e1_info.append(empty_list)
            e2_info.append(empty_list)
            e1info_attention_mask.append(ini_attention_mask)
            e2info_attention_mask.append(ini_attention_mask)
            # 外部资源
            for line in lines:
                e1_ok = 0
                e2_ok = 0
                e1_id = str(e1_arr[i])
                w1, w2 = line.strip('\n').split('\t')
                # w1, w2, max_len, head = lines2.strip('\n').split('\t')

                if e1_id == w1 and e1_ok == 0:
                    know = w2.replace('[', '').replace(']', '')
                    know = know.split(',')
                    c = list(map(int, know))

                    aa = pd.DataFrame(c).replace(0, np.NAN)
                    aa.dropna(inplace=True)
                    aa = np.array(aa).astype(np.int32).reshape(1, -1)[0]
                    aa = aa.tolist()

                    e_mask_len = len(aa)
                    e1_info[i] = c
                    e1info_attention_mask[i] = [1] * e_mask_len + [0] * (400 - e_mask_len)
                    e1_ok = 1

                e2_id = str(e2_arr[i])
                if e2_id == w1 and e2_ok == 0:
                    know2 = w2.replace('[', '').replace(']', '')
                    know2 = know2.split(',')
                    d = list(map(int, know2))

                    aa = pd.DataFrame(d).replace(0, np.NAN)
                    aa.dropna(inplace=True)
                    aa = np.array(aa).astype(np.int32).reshape(1, -1)[0]
                    aa = aa.tolist()
                    e_mask_len = len(aa)
                    e2info_attention_mask[i] = [1] * e_mask_len + [0] * (400 - e_mask_len)
                    e2_info[i] = d
                    e2_ok = 1

                if e1_ok == 1 and e2_ok == 1:
                    break

        f_extra.close()

        # 把得到的信息送入模型(之后把attention换成e1attention，e2attention)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 需要先变成tensor（）
        e1_info = torch.LongTensor(e1_info)
        e2_info = torch.LongTensor(e2_info)
        e1info_attention_mask = torch.LongTensor(e1info_attention_mask)
        e2info_attention_mask = torch.LongTensor(e2info_attention_mask)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        e1_info = e1_info.to(device)
        e2_info = e2_info.to(device)
        e1info_attention_mask = e1info_attention_mask.to(device)
        e2info_attention_mask = e2info_attention_mask.to(device)

        e1_outputs = self.bert(e1_info, e1info_attention_mask, token_type_ids=token_type_ids)
        e2_outputs = self.bert(e2_info, e2info_attention_mask, token_type_ids=token_type_ids)

        e1_sequence_output = e1_outputs[0]
        e2_sequence_output = e2_outputs[0]
        e1_pooled_output = e1_outputs[1]
        e2_pooled_output = e2_outputs[1]

        # 不参加实体平均
        # Dropout -> tanh -> fc_layer
        pooled_output = self.cls_fc_layer(pooled_output)

        # 与cls一样 information
        e1_pooled_output = self.cls_fc_layer(e1_pooled_output)
        e2_pooled_output = self.cls_fc_layer(e2_pooled_output)

        e1_h = self.e1_fc_layer(e1_h)
        e2_h = self.e2_fc_layer(e2_h)

        # Concat -> fc_layer
        lianhe1 = e1_h + e1_pooled_output * 0.4
        lianhe2 = e2_h + e2_pooled_output * 0.4

        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        concat_h2 = torch.cat([pooled_output, lianhe1, lianhe2], dim=-1)
        # m1 = nn.LayerNorm(concat_h.size()[0:]).cuda()
        # concat_h = m1(concat_h)

        # concat_h=self.out_layer(concat_h)  # dropout
        # m2 = nn.LayerNorm(concat_h2.size()[0:]).cuda()
        # concat_h2 = m2(concat_h2)
        # concat_h2 = self.out_layer(concat_h2)  # dropout

        logits = self.label_classifier(concat_h)
        logits2 = self.label_classifier(concat_h2)
        lianhe = logits * 0.5 + logits2 * 0.5

        # 1.拆分2.合并
        # gcn_outputs = [[0] * 400 for i in range(batch)]

        # e1_gcn_outputs = torch.LongTensor(e1_gcn_outputs)
        # e2_gcn_outputs = torch.LongTensor(e2_gcn_outputs)
        # gcn_out1 = e1_h
        # gcn_out2 = e2_h
        # gcn_input=sequence_output
        # gcn_input2=sequence_output
        # label_gcn=labels
        head_list = li22
        # gcn_tensor=torch.Tensor(0)

        # config_s = BertConfig()
        # config_s2 = BertConfig
        # att1 = BertAttention(config_s).to(device)
        # att_s=torch.unsqueeze(sequence_output,1)
        # att_s2 = torch.unsqueeze(e1_h, 1)
        m1_selfatt_out = self.att1(sequence_output, attention_mask=None, head_mask=None)
        m1_selfatt_out = m1_selfatt_out[0]
        #m1_selfatt_out = sequence_output*m1_selfatt_out

        #111111111111111111111111
        #m1_selfatt_out = sequence_output
        #1111111111111111111111111

        # m1_selfatt_out = att1(att_s2, attention_mask=None, head_mask=None)
        # label=labels

        # for i in range(batch):
        #     # 信息不为空
        #     # e1_gcn_outputs[i] = train.grap_h(e1_sequence_output[i], e1_head_list[i], e1_max_len_list[i])
        #     if flag==0:
        #         # train
        #         temp = (train.grap_h(m1_selfatt_out[i], head_list[i], 400, label)).t()#5*400
        #     else:
        #         #test
        #         temp = (train.grap_h2(m1_selfatt_out[i], e1_head_list[i], 400, label)).t()#5*400
        #     gcn_tensor=torch.mean(temp,0)
        #     gcn_tensor=self.gcn_fc_layer(gcn_tensor)

        # concat_h3_1=lianhe1
        # concat_h3_2=lianhe2

        # gcn_tensor=gcn_outputs[0]
        # gcn1_tensor = torch.Tensor(a).t()
        # gcn_tensor = gcn_tensor.t()
        # m3 = nn.LayerNorm(gcn_tensor.size()[0:]).cuda()
        # gcn_tensor = m3(gcn_tensor)

        # concat_h3_1 = concat_h3_1 + gcn_tensor*0.01

        # if e2_head_list[0]:
        #     gcn2_tensor = e2_gcn_outputs[0].t()
        #     m22 = nn.LayerNorm(gcn2_tensor.size()[0:]).cuda()
        #     gcn2_tensor = m22(gcn2_tensor)
        #     concat_h3_2 = concat_h3_2 + gcn2_tensor*0.001

        # aaa = e1_gcn_outputs[0]
        # b = e1_gcn_outputs[1]
        # c = e1_gcn_outputs[2] # batch=4
        # d = e1_gcn_outputs[3]
        # print(a)
        # gcn1_tensor = torch.stack(aaa, 1).t()
        # aa = e1_gcn_outputs[0]
        # bb = e1_gcn_outputs[1]
        # cc = e1_gcn_outputs[2]
        # dd = e1_gcn_outputs[3]
        # print(a)
        # gcn2_tensor = torch.stack([aa, bb, cc, dd], 1).t()
        # gcn2_tensor = torch.stack([aa], 1).t()
        # gcn_tensor=torch.unsqueeze(gcn_tensor,0)

        # 有待调试！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # concat_h3 = torch.cat([pooled_output, lianhe1, lianhe2,gcn_tensor*0.01], dim=-1)
        #
        # concat_h3 = torch.cat([pooled_output, lianhe1, lianhe2,gcn_tensor*0.1], dim=-1)
        # m3 = nn.LayerNorm(concat_h3.size()[0:]).cuda()
        # concat_h3 = m3(concat_h3)
        # lianhegcn = self.label_classifier2(concat_h3)
        # #lianhegcn = lianhe*0.5 + gcn_tensor*0.05

        # m1_selfatt_out = att1(sequence_output, attention_mask=attention_mask, head_mask=None)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # outputs2 = (logits2,) + outputs[2:]  # add hidden states and attention if they are here
        # outputs3 = (lianhe,) + outputs[2:]  # add hidden states and attention if they are here
        # outputs3 = (lianhegcn,) + outputs[2:]

        # f_count=open('./count.txt','a')
        # flag=0
        # # Softmax
        # if labels is not None:
        #     if self.num_labels == 1:
        #         loss_fct = nn.MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = nn.CrossEntropyLoss()
        #
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #         loss2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
        #         loss3 = loss_fct(lianhe.view(-1, self.num_labels), labels.view(-1))
        #         #loss3 = loss_fct(lianhegcn.view(-1, self.num_labels), labels.view(-1))
        #
        #         if loss2.item() < loss.item():
        #             if loss3.item() < loss2.item():
        #                 loss = loss3
        #                 outputs = outputs3
        #                 f_count.write('count3')
        #                 f_count.write('\n')
        #                 flag=1
        #             else:
        #                 loss = loss2
        #                 outputs = outputs2
        #                 f_count.write('count2')
        #                 f_count.write('\n')
        #                 flag = 1
        #         else:
        #             if loss3.item() < loss.item():
        #                 loss = loss3
        #                 outputs = outputs3
        #                 f_count.write('count3')
        #                 f_count.write('\n')
        #                 flag=1
        #         # if loss4.item()<loss.item():
        #         #     loss=loss4
        #         #     outputs=outputs4
        #     if flag==0:
        #         f_count.write('count')
        #         f_count.write('\n')
        #     f_count.write(str(labels.item()))
        #     outputs = (loss,) + outputs
        # f_count.close()

        # 400*768

        return m1_selfatt_out[
                   0], outputs, logits, logits2, lianhe, head_list, labels, concat_h2  # (loss), logits, (hidden_states), (attentions)
