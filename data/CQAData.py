# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import numpy as np


class CQAData(Dataset):

    def __init__(self, dataset, mode):

        self.a_qid = np.load(f"{dataset}/a_qid.npy", allow_pickle=True).tolist()
        self.q_title = np.load(f'{dataset}/q_title.npy', allow_pickle=True).tolist()
        self.q_body = np.load(f'{dataset}/q_body.npy', allow_pickle=True).tolist()
        self.q_tag = np.load(f'{dataset}/q_tag.npy', allow_pickle=True).tolist()

        if mode == 'Train':
            path = f'{dataset}/train'
        elif mode == 'Test':
            path = f'{dataset}/test'
        else:
            path = f'{dataset}/dev'

        self.qid_list = np.load(f'{path}/qid_list.npy', allow_pickle=True).tolist()
        self.aid_list = np.load(f'{path}/aid_list.npy', allow_pickle=True).tolist()
        self.label_list = np.load(f'{path}/label_list.npy', allow_pickle=True).tolist()

    def __getitem__(self, idx):
        assert idx < len(self)
        
        q_id = self.qid_list[idx]
        a_id = self.aid_list[idx]

        label = self.label_list[idx]

        q_body = self.q_body[q_id]
        q_title = self.q_title[q_id]
        q_tag = self.q_tag[q_id]

        aid_qid_list = self.a_qid[a_id]

        aid_q_titles = [self.q_title[q] for q in aid_qid_list]
        aid_q_bodys = [self.q_body[q] for q in aid_qid_list]
        aid_q_tags = [self.q_tag[q] for q in aid_qid_list]

        x = [label, q_id, a_id, q_title, q_body, q_tag, \
            aid_q_titles, aid_q_bodys, aid_q_tags]
        return x

    def __len__(self):
        return len(self.label_list)
