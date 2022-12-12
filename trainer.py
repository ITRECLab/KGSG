import os
import logging
from tqdm import tqdm, trange
from pygcn import train
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F

from model import RBERT
from utils import set_seed, compute_metrics, get_label, compute_metrics_test
from pygcn.models import GCN
from pygcn.train import my_train
from model import FCLayer

import torch.nn as nn
logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, args2, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.args2 = args2
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.label_lst = get_label(args)
        self.num_labels = len(self.label_lst)

        # 加载模型配置参数隐层数、隐层维度、激活函数and so on
        self.bert_config = BertConfig.from_pretrained(args.pretrained_model_name, num_labels=self.num_labels,
                                                      finetuning_task=args.task)
        # 通过配置参数加载模型
        self.model = RBERT(self.bert_config, args)
        self.model_g=None
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)
        #self.gcntensor = FCLayer(400, 768, args.dropout_rate)
        self.gcntensor = FCLayer(400, 768, args.dropout_rate)
        self.gcntensor.to(self.device)
        self.optimizer2=None
        self.feature_s = FCLayer(self.bert_config.hidden_size * 3, self.bert_config.hidden_size, args.dropout_rate)
        self.feature_g = FCLayer(self.bert_config.hidden_size, self.bert_config.hidden_size, args.dropout_rate)

        self.label_classifier2 = FCLayer(self.bert_config.hidden_size * 4, self.bert_config.num_labels, args.dropout_rate,
                                        use_activation=False)
        self.label_classifier3 = FCLayer(self.bert_config.hidden_size * 2, self.bert_config.num_labels, args.dropout_rate,
                                        use_activation=False)
        self.feature_s.to(self.device)
        self.feature_g.to(self.device)
        self.label_classifier3.to(self.device)
        # 可更新参数
        # self.bert_gcn_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.bert_gcn_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def train(self):
        # 打乱数据集
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                    len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)

        loss_holder = []
        loss_value = np.inf

        f = open('./loss.txt', 'a')
        flag = 0
        flag_gcn=0
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3],
                          'e1_mask': batch[4],
                          'e2_mask': batch[5],
                          'li': batch[6],
                          'li2': batch[7],
                          'flag': flag,
                          }
                #m1_selfatt_out,outputs,outputs2,outputs3,e1_head_list, labels,concat_h2 = self.model(**inputs)
                m1_selfatt_out,outputs,logits,logits2,lianhe,e1_head_list, labels, concat_h2 = self.model(**inputs)

                if flag_gcn==0:
                    self.model_g = GCN(nfeat=m1_selfatt_out.shape[1],
                                  nhid=self.args2.hidden,
                                  nclass=5,
                                  dropout=self.args2.dropout)
                    self.optimizer2 = torch.optim.Adam(self.model_g.parameters(),
                                            lr=self.args2.lr, weight_decay=self.args2.weight_decay)
                    scheduler2 = get_linear_schedule_with_warmup(self.optimizer2, num_warmup_steps=self.args.warmup_steps,
                                                                num_training_steps=t_total)

                    self.model_g.to(self.device)
                    #self.model_g.train()
                    flag_gcn=1
                adj,features = my_train(self.args2, m1_selfatt_out, e1_head_list, 400)


                gcn_tensor_output=self.model_g(features,adj).t()#5*400


                #gcn_tensor_output=(train.grap_h(m1_selfatt_out, e1_head_list, 400, label)).t()  # 5*400
                gcn_tensor_output=torch.mean(gcn_tensor_output,0)
                gcn_tensor_output=gcn_tensor_output.to(self.device)
                gcn_tensor_output=self.gcntensor(gcn_tensor_output)
                gcn_tensor_output=torch.unsqueeze(gcn_tensor_output,0)
                # # 有待调试！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                # m3 = nn.LayerNorm(gcn_tensor_output.size()[0:]).cuda()
                # gcn_tensor_output = m3(gcn_tensor_output)

                concat_h2=self.feature_s(concat_h2)
                #gcn_tensor_output=self.feature_g(gcn_tensor_output)

                concat_h3 = torch.cat([concat_h2, gcn_tensor_output], dim=-1)
                m4 = nn.LayerNorm(concat_h3.size()[0:]).cuda()
                concat_h3 = m4(concat_h3)
                lianhegcn = self.label_classifier3(concat_h3)
                # outputs2 = (logits2,) + outputs[2:]  # add hidden states and attention if they are here
                # outputs3 = (lianhe,) + outputs[2:]  # add hidden states and attention if they are here
                outputs = (lianhegcn,) + outputs[2:]

                # f_count=open('./count.txt','a')
                # f_count_num=0
                # Softmax
                if labels is not None:
                    if self.num_labels == 1:
                        loss_fct = nn.MSELoss()
                        loss = loss_fct(lianhegcn.view(-1), labels.view(-1))
                    else:
                        loss_fct = nn.CrossEntropyLoss()

                        loss = loss_fct(lianhegcn.view(-1, self.num_labels), labels.view(-1))

                    outputs = (loss,) + outputs
                # f_count.write(str(f_count_num))
                # f_count.close()
                loss = outputs[0]
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                    print(2)

                loss.backward()
                loss_save = str(loss.item())
                f.write(loss_save)
                f.write('\n')

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    # 只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整
                    optimizer.step()  # 单次优化 所有的optimizer都实现了step()方法，这个方法会更新所有的参数
                    self.optimizer2.step()
                    scheduler.step()  # Update learning rate schedule
                    scheduler2.step()  # Update learning rate schedule

                    self.model.zero_grad()  # 梯度清零
                    self.model_g.zero_grad()  # 梯度清零
                    global_step += 1  #

                    if global_step % 100 == 0:
                        loss_holder.append([global_step, loss])

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        #self.evaluate('test')
                        pass

                    # loss比较
                    #if self.args.save_steps > 0 and global_step % self.args.save_steps == 0 and loss < loss_value:
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        # print(loss, loss_value)
                        # loss_value = loss
                        self.save_model()
                        #torch.save(self.model_g, './model_g/training_config.pt')
                        #torch.save(self.model_g.state_dict(), './model_g/training_config.pt')
                    # if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    #     #pass
                    #     self.evaluate('test')
                        #pass
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        self.evaluate('test')
        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == 'test':
            dataset = self.test_dataset

        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")
        # train_sampler = RandomSampler(self.train_dataset)
        # train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None


        self.model.eval()
        self.model_g.eval()
        flag = 1
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3],
                          'e1_mask': batch[4],
                          'e2_mask': batch[5],
                          'li': batch[6],
                          'li2': batch[7],
                          'flag': flag,
                          }
                #outputs = self.model(**inputs)
                m1_selfatt_out, outputs, logits, logits2, lianhe, e1_head_list, labels, concat_h2 = self.model(**inputs)
                adj, features = my_train(self.args2, m1_selfatt_out, e1_head_list, 400)
                gcn_tensor_output = self.model_g(features, adj).t()  # 5*400

                # gcn_tensor_output=(train.grap_h(m1_selfatt_out, e1_head_list, 400, label)).t()  # 5*400
                gcn_tensor_output = torch.mean(gcn_tensor_output, 0)
                gcn_tensor_output = gcn_tensor_output.to(self.device)
                gcn_tensor_output = self.gcntensor(gcn_tensor_output)
                gcn_tensor_output = torch.unsqueeze(gcn_tensor_output, 0)

                concat_h2=self.feature_s(concat_h2)
                #gcn_tensor_output=self.feature_g(gcn_tensor_output)
                # # 有待调试！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                # m3 = nn.LayerNorm(gcn_tensor_output.size()[0:]).cuda()
                # gcn_tensor_output = m3(gcn_tensor_output)

                # mrs yang
                # ent_rep_s = torch.matmul(concat_h2, self.weight_1) + self.bias_1  # [bs, max_ent_cnt, config.hidden_size]
                #
                # context_atts_s = torch.matmul(gcn_tensor_output,
                #                               self.weight_2) + self.bias_2  # [bs, max_ent_cnt, max_length]
                # simi = torch.sigmoid(torch.cat([ent_rep_s, context_atts_s], -1))  # [bs, max_ent_cnt, 512+768]
                # simi = self.linear(simi)
                # gcn_tensor_output = torch.tanh(torch.matmul(gcn_tensor_output, self.weight_3) + self.bias_3)
                # # [bs, max_ent_cnt, max_length]
                # last_context_atts = torch.matmul(simi, gcn_tensor_output.permute(0, 2, 1))
                # concat_h2 = self.ent_trans(concat_h2)
                # last_context_atts = self.context_att_trans(last_context_atts)
                # concat_h2 = torch.cat([concat_h2, last_context_atts], dim=-1)
                # concat_h2 = torch.relu_(concat_h2)
                # concat_h2 = self.dim_trans(concat_h2)
                concat_h3 = torch.cat([concat_h2, gcn_tensor_output], dim=-1)
                m4 = nn.LayerNorm(concat_h3.size()[0:]).cuda()
                concat_h3 = m4(concat_h3)
                # w11 = self.bert_gcn_weight_1
                # w22 = self.bert_gcn_weight_2
                # w11 = w11.to(self.device)
                # w22 = w22.to(self.device)
                # concat_h3 = torch.cat([concat_h2 * w11, gcn_tensor_output * w22], dim=-1)
                # print(self.bert_gcn_weight_1.device, self.bert_gcn_weight_2.device)
                # print(self.bert_gcn_weight_1,self.bert_gcn_weight_2)


                #concat_h3 = torch.cat([concat_h2, gcn_tensor_output], dim=-1)
                lianhegcn = self.label_classifier3(concat_h3)
                # outputs2 = (logits2,) + outputs[2:]  # add hidden states and attention if they are here
                # outputs3 = (lianhe,) + outputs[2:]  # add hidden states and attention if they are here
                outputs = (lianhegcn,) + outputs[2:]

                # Softmax
                if labels is not None:
                    if self.num_labels == 1:
                        loss_fct = nn.MSELoss()
                        loss = loss_fct(lianhegcn.view(-1), labels.view(-1))
                    else:
                        loss_fct = nn.CrossEntropyLoss()

                        loss = loss_fct(lianhegcn.view(-1, self.num_labels), labels.view(-1))

                    outputs = (loss,) + outputs

                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # softmax_1 = F.softmax(logits, dim=1)
            # softmax_2 = F.softmax(logits2, dim=1)
            # softmax_3 = F.softmax(lianhe, dim=1)
            # softmax_4 = F.softmax(lianhegcn, dim=1)
            # # np.amax()
            # a = softmax_1.max(axis=1)
            # b = softmax_2.max(axis=1)
            # c = softmax_3.max(axis=1)
            # d = softmax_4.max(axis=1)
            # a_max = max(a, b, c, d)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                # preds2 = logits2.detach().cpu().numpy()
                # preds3 = lianhe.detach().cpu().numpy()
                # preds4 = lianhegcn.detach().cpu().numpy()
                # if a_max==a:
                #     preds=preds
                # if a_max==b:
                #     preds=preds2
                # if a_max==c:
                #     preds=preds3
                # if a_max==d:
                #     preds=preds4
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                # f_max=0
                # if a_max==a and f_max==0:
                #     preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                #     f_max=1
                #
                # if a_max==b and f_max==0:
                #     preds = np.append(preds, logits2.detach().cpu().numpy(), axis=0)
                #     f_max=1
                #
                # if a_max==c and f_max==0:
                #     preds = np.append(preds, lianhe.detach().cpu().numpy(), axis=0)
                #     f_max=1
                #
                # if a_max==d and f_max==0:
                #     preds = np.append(preds, lianhegcn.detach().cpu().numpy(), axis=0)
                #     f_max=1
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        preds = np.argmax(preds, axis=1)
        np.savetxt("./preds.txt", preds,fmt='%f')
        np.savetxt("./true.txt", out_label_ids,fmt='%f')
        if mode=='dev':
            result = compute_metrics(preds, out_label_ids)
        else:
            result = compute_metrics_test(preds, out_label_ids)
        results.update(result)
        f = open('./eval.txt', 'a')
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            f.writelines(key)
            f.write('\t')
            f.writelines(str(results[key]))
            f.write('\n')
        f.close()
        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model path doesn't exists! ")
        output_dir = os.path.join(self.args.model_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_config.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.bert_config = BertConfig.from_pretrained(self.args.model_dir)
            logger.info("***** Bert config loaded *****")
            self.model = RBERT.from_pretrained(self.args.model_dir, config=self.bert_config, args=self.args)
            self.model.to(self.device)
            #self.model_g = torch.load('./model_g/training_config.pt')
            #self.model_g = torch.load('./model_g/training_config.pt')
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
