import json
import glob
import csv
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import svm, tree
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, scale
from scipy.cluster.vq import whiten
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import re
import os

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch
import math
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.model_selection import StratifiedKFold

import ass



def get_optimizers(model, learning_rate, adam_epsilon, weight_decay, num_training_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    # optimizer = SGD(optimizer_grouped_parameters, lr=learning_rate, momentum=0.9)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=num_training_steps)
    return optimizer, scheduler


class DNNAudio(nn.Module):
    def __init__(self):
        super(DNNAudio, self).__init__()
        self.dim = 256
        self.layer1 = nn.Linear(88, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)
        # self.layer2 = nn.Linear(256, 1)
        self.layer3 = nn.Linear(self.dim, self.dim)
        # self.layer3 = nn.Linear(256, 1)
        # self.layer4 = nn.Linear(self.dim, 1)
        self.layer4 = nn.Linear(self.dim, self.dim)
        # self.ln = nn.LayerNorm(256)
        # self.bn = nn.BatchNorm1d(256)

    def forward(self, input_features):
        # return F.relu(self.layer2(F.relu(self.layer1(input_features))))
        # return self.layer3(F.relu(self.layer2(F.relu(self.layer1(input_features)))))
        return self.layer4(F.relu(self.layer3(F.relu(self.layer2(F.relu(self.layer1(input_features)))))))


class config:
    mode = 'regression'

class jointTAMulti(nn.Module):
    def __init__(self, TRIconfig):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.text_emb = nn.Linear(768, 256)
        self.mode = TRIconfig.mode
        self.AudioNet = DNNAudio()
        self.regress = nn.Linear(512, 1)
        self.classify = nn.Linear(512, 2)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                labels,
                audio_features):
        audio_repr = self.AudioNet(audio_features)
        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        text_repr = self.text_emb(bert_outputs[1])
        final_repr = torch.cat([audio_repr, text_repr], dim=1)
        final_repr = self.dropout(final_repr)
        # logits = self.classifier(final_repr)
        # print(logits, labels)
        if self.mode == 'regression':
            logits = self.regress(final_repr)
            loss_fct = nn.MSELoss()
            out_loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            logits = self.classify(final_repr)
            loss_fct = nn.CrossEntropyLoss()
            out_loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return out_loss, logits

    def set_mode(self, mode):
        self.mode = mode


#=====================================================================================================================
with open('all_data_joint_fixed_single_task.pkl','rb') as dump_file:
    audios, activation_labels, valence_labels, stress_labels, input_ids, token_type_ids, attention_mask, cv5_ids = pkl.load(dump_file)

print(audios.shape)
print(activation_labels)
print(valence_labels)
print(stress_labels)

sp = cv5_ids[2]

tv_a, tv_l, tv_labels = audios[sp[0]], input_ids[sp[0]], stress_labels[sp[0]]
train_num = len(tv_a) - 200
train_a, train_l, train_labels = tv_a[:train_num], tv_l[:train_num], tv_labels[:train_num]
dev_a, dev_l, dev_labels = tv_a[train_num:], tv_l[train_num:], tv_labels[train_num:]
test_a, test_l, test_labels = audios[sp[1]], input_ids[sp[1]], stress_labels[sp[1]]

train_activ, dev_activ, test_activ, train_valence, dev_valence, test_valence = activation_labels[sp[0]][:train_num], \
    activation_labels[sp[0]][train_num:], activation_labels[sp[1]], valence_labels[sp[0]][:train_num], \
    valence_labels[sp[0]][train_num:], valence_labels[sp[1]],

tv_token_type_ids, test_token_type_ids, tv_attention_mask, test_attention_mask = token_type_ids[sp[0]], \
                                           token_type_ids[sp[1]], attention_mask[sp[0]], attention_mask[sp[1]]

train_token_type_ids, train_attention_mask = tv_token_type_ids[:train_num], tv_attention_mask[:train_num]
dev_token_type_ids, dev_attention_mask = tv_token_type_ids[train_num:], tv_attention_mask[train_num:]

n_train = len(train_a)
n_dev = len(dev_a)
n_test = len(test_a)


print(train_l.shape, train_a.shape)

TRIconfig = config()
TRIconfig.mode = 'classify'

# to Tensors
train_labels, dev_labels, test_labels = torch.LongTensor(train_labels), torch.LongTensor(dev_labels), \
                                            torch.LongTensor(test_labels)
train_activ, dev_activ, test_activ, train_valence, dev_valence, test_valence = \
    torch.FloatTensor(train_activ), torch.FloatTensor(dev_activ), torch.FloatTensor(test_activ), \
    torch.FloatTensor(train_valence), torch.FloatTensor(dev_valence), torch.FloatTensor(test_valence)

train_a, dev_a, test_a = torch.FloatTensor(train_a), torch.FloatTensor(dev_a), torch.FloatTensor(test_a)

train_l, dev_l, test_l, train_token_type_ids, dev_token_type_ids, test_token_type_ids = torch.LongTensor(train_l), \
                                                             torch.LongTensor(dev_l), \
                                                             torch.LongTensor(test_l), \
                                                             torch.LongTensor(train_token_type_ids), \
                                                             torch.LongTensor(dev_token_type_ids), \
                                                             torch.LongTensor(test_token_type_ids)

train_attention_mask, dev_attention_mask, test_attention_mask = torch.FloatTensor(train_attention_mask), \
                                            torch.FloatTensor(dev_attention_mask), \
                                            torch.FloatTensor(test_attention_mask)


model = jointTAMulti(TRIconfig).to('cuda')

eval_every = 5
batch_size = 32
test_batch_size = 4
max_epochs = 1000
t_total = math.ceil(n_train / batch_size) * max_epochs
lr = 3e-4
epsilon = 1e-8
max_grad_norm = 1.0
weight_decay = 0.0

optimizer, scheduler = get_optimizers(model, learning_rate=lr, adam_epsilon=epsilon, weight_decay=weight_decay,
                                      num_training_steps=t_total)

# loss_fn = torch.nn.CrossEntropyLoss().cuda()
model.train()
model.zero_grad()
# pre-training
pre_train_epoch = 0
# model.set_mode('regression')
aux_task = "val"

for ep in range(max_epochs):
    total_samples = 0
    avg_loss = 0
    n_batch = 0
    model.train()
    while total_samples < n_train:
        optimizer.zero_grad()
        selected_id = np.random.permutation(n_train)[:batch_size]
        batch_a = train_a[selected_id].to('cuda')
        batch_l = train_l[selected_id].to('cuda')

        batch_ty = train_token_type_ids[selected_id].to('cuda')
        batch_am = train_attention_mask[selected_id].to('cuda')
        # switch between modes uniformly
        selected_task = np.random.randint(low=0, high=2)
        if selected_task == 0:
            ans = train_valence[selected_id].to('cuda')
            model.set_mode('regression')
        else:
            ans = train_labels[selected_id].to('cuda')
            model.set_mode('classify')
        '''elif selected_task == 1:
                        ans = train_valence[selected_id].to('cuda')
                        model.set_mode('regression')'''

        total_samples += batch_size
        loss, logits = model(input_ids=batch_l, token_type_ids=batch_ty, attention_mask=batch_am, labels=ans, audio_features=batch_a)
        # print(loss)
        # logits = torch.squeeze(logits, dim=1)
        # print(preds[0], preds[1])
        # print(preds.shape, ans.shape)
        # print(preds, ans)
        loss.backward()
        # print(loss.data.cpu().numpy())
        avg_loss += loss.data.cpu().numpy()
        n_batch += 1.

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

        torch.cuda.empty_cache()

    del batch_l, batch_ty, batch_am, ans, batch_a
    torch.cuda.empty_cache()
    avg_loss = avg_loss / n_batch
    print("epoch: %d rmse/ce: %f" % (ep + 1, avg_loss ** 0.5))


    # time.sleep(20)

    if ep % eval_every == 0:
        idx = 0
        model.set_mode('classify')
        model.eval()
        total_loss = 0.
        eval_preds = np.array([])
        n_batch = 0
        while idx < n_test:
            test_batch_a = test_a[idx:(idx + test_batch_size)].to('cuda')

            test_batch_l = test_l[idx:(idx + test_batch_size)].to('cuda')
            test_batch_ty = test_token_type_ids[idx:(idx + test_batch_size)].to('cuda')
            test_batch_am = test_attention_mask[idx:(idx + test_batch_size)].to('cuda')
            # time.sleep(20)
            # exit()
            if TRIconfig.mode == 'regression':
                test_ans = test_activ[idx:(idx + test_batch_size)].to('cuda') if aux_task == 'act' else test_valence[
                                                                            idx:(idx + test_batch_size)].to('cuda')
            else:
                test_ans = test_labels[idx:(idx + test_batch_size)].to('cuda')

            loss, logits = model(input_ids=test_batch_l,
                                 token_type_ids=test_batch_ty,
                                 attention_mask=test_batch_am,
                                 labels=test_ans,
                                 audio_features=test_batch_a)
            if TRIconfig.mode == 'regression':
                mse = loss.data.cpu().numpy()
                total_loss += mse
            else:
                _, batch_eval_preds = logits.data.cpu().max(1)
                # print(batch_eval_preds, test_ans)
                eval_preds = np.concatenate((eval_preds, batch_eval_preds), axis=-1)
            # test_pred = torch.squeeze(test_pred, dim=1)
            if idx % 50 == 0:
                print(logits, test_ans)

            idx += test_batch_size
            torch.cuda.empty_cache()
            n_batch += 1.

        del test_batch_l, test_batch_ty, test_batch_am, test_ans, test_batch_a
        torch.cuda.empty_cache()
        # metrics
        if TRIconfig.mode == 'regression':
            print('evaluation rmse: %f' % (total_loss / n_batch) ** 0.5)
        else:
            precison, recall, fscore, support = precision_recall_fscore_support(test_labels.cpu().numpy(), eval_preds,
                                                                                labels=[1], average=None)
            # print('saving:')
            print(float(sum(eval_preds == test_labels.cpu().numpy())) / len(eval_preds))
            print(precison, recall, fscore, support)
        # print('saving:')

        '''model_dir = save_dir + '%d' % (ep+1)
        os.mkdir(model_dir)
        model.save_pretrained(model_dir)'''