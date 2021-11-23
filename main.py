# -*- encoding: utf-8 -*-
import time
import random
import fire

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import CQAData
import config
from utils import cal_metric, group_resuts
import models

import tokenizers


tokenizer = tokenizers.Tokenizer.from_file('./data/EnglistByte.token')
opt = getattr(config, 'Bioinformatics_Config')()


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def token_encode(str_list):
    str_list = tokenizer.encode_batch(str_list)
    ids = [i.ids for i in str_list]
    mask = [i.attention_mask for i in str_list]

    return torch.tensor(ids).cuda(), torch.tensor(mask).cuda()


def batch_token_encode(q_str_list_batch):
    all_titles = []
    BS = len(q_str_list_batch)
    K = len(q_str_list_batch[0])
    for i in range(BS):
        q_str_list = q_str_list_batch[i]
        for q in q_str_list:
            all_titles.append(q)

    all_ids, all_masks = token_encode(all_titles)
    all_ids = all_ids.view(BS, K, -1)
    all_masks = all_masks.view(BS, K, -1)
    return all_ids, all_masks


def collate_fn(batch):
    '''
    1. Tokenize the text here
    2. to Pytorch Tensor from list/ndaary
    '''
    label, q_id, a_id, q_title, q_body, q_tag, aid_q_titles, aid_q_bodys, aid_q_tags = zip(*batch)

    label = torch.tensor(label).cuda()
    q_id = torch.tensor(q_id).cuda()
    a_id = torch.tensor(a_id).cuda()
    q_tag = torch.tensor(q_tag).cuda()
    aid_q_tags = torch.tensor(aid_q_tags).cuda()

    q_title_id, q_title_mask = token_encode(q_title)
    q_body_id, q_body_mask = token_encode(q_body)
    aid_q_titles_id, aid_q_titles_mask = batch_token_encode(aid_q_titles)
    aid_q_bodys_id, aid_q_bodys_mask = batch_token_encode(aid_q_bodys)

    return label, q_id, a_id, q_title_id, q_title_mask, q_body_id, q_body_mask, q_tag, \
             aid_q_titles_id, aid_q_titles_mask, aid_q_bodys_id, aid_q_bodys_mask, aid_q_tags


def run(**kwargs):

    global opt
    if 'dataset' in kwargs:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    torch.cuda.set_device(opt.gpu_id)

    model = getattr(models, opt.model)(opt).cuda()

    train_data = CQAData(opt.data_path, mode="Train")
    test_data = CQAData(opt.data_path, mode="Test")
    dev_data = CQAData(opt.data_path, mode="Dev")

    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    dev_data_loader = DataLoader(dev_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f'train data: {len(train_data)}; test data: {len(test_data)}; dev data: {len(dev_data)}')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    bce = nn.BCEWithLogitsLoss()
    min_p1 = 1e-10
    print("start training....")

    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        model.train()
        for idx, train_datas in enumerate(train_data_loader):
            label, _, data = train_datas[0], train_datas[1], train_datas[2:]
            optimizer.zero_grad()
            out = model(data)
            loss = bce(out.squeeze(1), label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        mean_loss = total_loss * 1.0 / idx
        print(f"{now()}  Epoch {epoch}: train data: loss:{mean_loss:.4f}.")

        print("dev data results")
        predict_p1 = dev(model, dev_data_loader, opt.metrics)

        if predict_p1 > min_p1:
            
            min_p1 = predict_p1
            ti = time.strftime('%Y%m%d_%H:%M:%S')
            name = 'checkpoints/' + 'EFHM' + '_' + 'best' + '.opt'
            torch.save(model.state_dict(), name)

    print("*****"*20)
    print("test data results")
    model.load_state_dict(torch.load(name))
    test(model, test_data_loader, opt.metrics)
    print("*****"*20)

def test(model, data_loader, metrics):

    model.eval()

    all_labels = []
    all_preds = []
    all_qid = []
    with torch.no_grad():
        for idx, test_data in enumerate(data_loader):
            label, q_id, data = test_data[0], test_data[1], test_data[2:]
            out = model(data).cpu().numpy()
            all_preds.extend(np.reshape(out, -1))
            all_labels.extend(label.cpu().numpy())
            all_qid.extend(q_id.cpu().numpy())
    all_labels, all_preds = group_resuts(all_labels, all_preds, all_qid)
    res = cal_metric(all_labels, all_preds, metrics)
    res = [f"{k}: {v:.4f};" for k, v in res.items()]
    print(' '.join(res))

def dev(model, data_loader, metrics):

    model.eval()
    all_labels = []
    all_preds = []
    all_qid = []
    with torch.no_grad():
        for idx, dev_data in enumerate(data_loader):
            label, q_id, data = dev_data[0], dev_data[1], dev_data[2:]
            out = model(data).cpu().numpy()
            all_preds.extend(np.reshape(out, -1))
            all_labels.extend(label.cpu().numpy())
            all_qid.extend(q_id.cpu().numpy())
    all_labels, all_preds = group_resuts(all_labels, all_preds, all_qid)
    res = cal_metric(all_labels, all_preds, metrics)
    predict_p1 = res['P@1']
    res = [f"{k}: {v:.4f};" for k, v in res.items()]
    print(' '.join(res))

    return predict_p1


if __name__ == "__main__":
    fire.Fire()