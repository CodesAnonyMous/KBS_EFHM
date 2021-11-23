# -*- coding: utf-8 -*-
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder


class Multi_Grained(nn.Module):
    '''
    Multi Grained
    '''
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.batch_size = opt.batch_size

        self.word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.answerer_embs = nn.Embedding(opt.a_num + 5, opt.word_dim)
        self.tag_embs = nn.Embedding(opt.tag_num + 5, opt.word_dim)

        self.q_title_encoder = Encoder(opt.enc_method, opt.word_dim, opt.fea_size*2, opt.fea_size)
        self.q_body_encoder = Encoder(opt.enc_method, opt.word_dim, opt.fea_size*2, opt.fea_size)
        self.h_title_encoder = Encoder(opt.enc_method, opt.word_dim, opt.fea_size*2, opt.fea_size)
        self.h_body_encoder = Encoder(opt.enc_method, opt.word_dim, opt.fea_size*2, opt.fea_size)

        self.f_fc = nn.Linear(opt.fea_size*2, opt.fea_size)
        self.predict = nn.Linear(2 * opt.fea_size, 1)
        self.dropout = nn.Dropout(opt.drop_out)

        self.q_title_fc = nn.Linear(opt.fea_size, opt.fea_size)
        self.q_body_fc = nn.Linear(opt.fea_size, opt.fea_size)
        self.q_fixFea_qv = nn.Linear(opt.fea_size, opt.fea_size)

        self.q_tag_fc = nn.Linear(opt.fea_size, opt.fea_size)
        self.aid_q_tags_fc = nn.Linear(opt.fea_size, opt.fea_size)

        self.H_body_qv = nn.Linear(opt.fea_size, opt.fea_size)
        self.H_title_qv = nn.Linear(opt.fea_size, opt.fea_size)
        self.H_fixFea_qv = nn.Linear(opt.fea_size, opt.fea_size)

        self.w_score = nn.Linear(1, 1, bias=False)
        self.t_score = nn.Linear(1, 1, bias=False)
        self.u_score = nn.Linear(1, 1, bias=False)

        self.word_linear = nn.Linear(opt.fea_size, opt.fea_size)
        self.title_linear = nn.Linear(opt.fea_size, opt.fea_size)
        self.user_linear = nn.Linear(opt.fea_size, opt.fea_size)

        self.reset_para()

    def forward(self, data):

        a_id, q_title_id, q_title_mask, q_body_id, q_body_mask, q_tag, \
            aid_q_titles_id, aid_q_titles_mask, aid_q_bodys_id, aid_q_bodys_mask, aid_q_tags = data

        # ---------------------------------------------------------------------

        answerer_emb = self.answerer_embs(a_id) 

        q_title_embs = self.word_embs(q_title_id)
        q_title_fea = self.q_title_encoder(q_title_embs) 
        q_title_fea = q_title_fea * q_title_mask.unsqueeze(2) 
        q_titlefea = self.q_title_fc(q_title_fea.mean(1))

        u_q_list_title = self.word_embs(aid_q_titles_id) 
        b, m, l, d = u_q_list_title.size() 

        u_q_list_title = u_q_list_title.view(-1, l, d)
        u_q_fea = self.h_title_encoder(u_q_list_title)
        u_q_fea_title = u_q_fea.mean(1)   
        H_title = u_q_fea_title.view(b, m, -1) 
        H_word = u_q_list_title.view(b, -1, d)

        H_title_qv = self.H_fixFea_qv(answerer_emb).unsqueeze(2)
        H_fixAttweight = torch.bmm(H_title, H_title_qv)
        H_fixScore = F.softmax(H_fixAttweight, 1)
        H_fixFea = H_title * H_fixScore
        u_fea = H_fixFea.sum(1)


        H_title_score = torch.bmm(self.dropout(H_title), self.dropout(self.word_linear(q_titlefea)).unsqueeze(2))
        H_title_score = torch.max(H_title_score, 1)[0]

        H_word_score = torch.bmm(self.dropout(H_word), self.dropout(self.title_linear(q_titlefea)).unsqueeze(2))
        H_word_score = torch.max(H_word_score, 1)[0]

        u_fea = torch.cat([self.dropout(u_fea), self.dropout(self.user_linear(q_titlefea))], 1)
        user_score = self.predict(u_fea)


        out = self.w_score(H_word_score) + self.t_score(H_title_score) + self.u_score(user_score)

        return out

    def reset_para(self):
        nn.init.uniform_(self.word_embs.weight, -0.2, 0.2)
        nn.init.uniform_(self.answerer_embs.weight, -0.2, 0.2)
        nn.init.uniform_(self.tag_embs.weight, -0.2, 0.2)

        fcs = [self.f_fc, self.predict, self.q_title_fc, self.q_body_fc, self.H_fixFea_qv, \
                self.H_title_qv, self.H_body_qv, self.q_fixFea_qv, self.word_linear, self.title_linear, self.user_linear]
        for fc in fcs:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.uniform_(fc.bias, 0.01)

        nn.init.xavier_uniform_(self.w_score.weight)
        nn.init.xavier_uniform_(self.t_score.weight)
        nn.init.xavier_uniform_(self.u_score.weight)
