# -*- coding: utf-8 -*-

import numpy as np


class DefaultConfig:

    model = 'Multi_Grained'
    dataset = 'Bioinformatics'

    # -------------base config-----------------------#
    gpu_id = 1
    gpu_ids = []

    seed = 2019
    num_epochs = 10
    num_workers = 0

    enc_method = 'transformer'

    optimizer = 'Adam'
    weight_decay = 5e-4 
    lr = 1e-3
    drop_out = 0.3

    metrics = ['mean_mrr', 'P@1', 'ndcg@10']

    CANDIDATE_NUM = 20
    word_dim = 100
    cell_length = 3
    vocab_size = 30000
    id_emb_size = 100
    fea_size = 100
    loss_weight = [2.0, 0.05, 0.001]
    pos_r = [1.0, 1.0]

    batch_size = 128

    def set_path(self, name):
        '''
        specific
        '''
        self.data_path = f'./data/{name}'

        self.answerer_id_path = f'{self.data_path}/aid.npy'
        self.answerer_history_path = f'{self.data_path}/a_history.npy'

    def parse(self, kwargs):

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


class AI_Config(DefaultConfig):

    def __init__(self):
        self.set_path('AI')
    dataset = 'AI'
    a_num = 195
    q_num = 1205
    tag_num = 595


class Bioinformatics_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Bioinformatics')
    dataset = 'Bioinformatics'
    a_num = 113
    q_num = 958
    tag_num = 435


class print_Config(DefaultConfig):

    def __init__(self):
        self.set_path('print')
    dataset = 'print'
    a_num = 112
    q_num = 1033
    tag_num = 377


class Biology_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Biology')
    dataset = 'Biology'
    a_num = 630
    q_num = 8704
    tag_num = 743


class English_Config(DefaultConfig):

    def __init__(self):
        self.set_path('English')
    dataset = 'English'
    a_num = 4781
    q_num = 46692
    tag_num = 971


class History_Config(DefaultConfig):

    def __init__(self):
        self.set_path('History')
    dataset = 'History'
    a_num = 471
    q_num = 4904
    tag_num = 811
