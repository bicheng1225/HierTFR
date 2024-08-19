# -*- coding: utf-8 -*-
# @Time    : 9/20/21 12:02 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

# train and test the models
import sys
import os
import time
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from models import *
import argparse
import dgl
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs_pretrain-mdl_v11_v2')
import torch
import numpy as np
import torch.nn as nn

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp-dir", type=str, default="./exp/", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--n_epochs", type=int, default=100, help="number of maximum training epochs")
parser.add_argument("--goptdepth", type=int, default=1, help="depth of gopt models")
parser.add_argument("--goptheads", type=int, default=1, help="heads of gopt models")
parser.add_argument("--batch_size", type=int, default=25, help="training batch size")
parser.add_argument("--embed_dim", type=int, default=12, help="gopt transformer embedding dimension")
parser.add_argument("--loss_w_phn", type=float, default=1, help="weight for phoneme-level loss")
parser.add_argument("--loss_w_word", type=float, default=1, help="weight for word-level loss")
parser.add_argument("--loss_w_utt", type=float, default=1, help="weight for utterance-level loss")
parser.add_argument("--model", type=str, default='gopt', help="name of the model")
parser.add_argument("--am", type=str, default='librispeech', help="name of the acoustic models")
parser.add_argument("--noise", type=float, default=0., help="the scale of random noise added on the input GoP feature")

# just to generate the header for the result.csv
def gen_result_header():
    phn_header = ['epoch', 'phone_train_mse', 'phone_train_pcc', 'phone_test_mse', 'phone_test_pcc', 'learning rate']
    utt_header_set = ['utt_train_mse', 'utt_train_pcc', 'utt_test_mse', 'utt_test_pcc']
    utt_header_score = ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']
    word_header_set = ['word_train_pcc', 'word_test_pcc']
    word_header_score = ['accuracy', 'stress', 'total']
    utt_header, word_header = [], []
    for dset in utt_header_set:
        utt_header = utt_header + [dset+'_'+x for x in utt_header_score]
    for dset in word_header_set:
        word_header = word_header + [dset+'_'+x for x in word_header_score]
    header = phn_header + utt_header + word_header
    return header

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))

    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_mse = 0, 999
    global_step, epoch = 0, 0
    exp_dir = args.exp_dir

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} k'.format(sum(p.numel() for p in audio_model.parameters()) / 1e3))
    print('Total trainable parameter number is : {:.3f} k'.format(sum(p.numel() for p in trainables) / 1e3))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(20, 100, 5)), gamma=0.5, last_epoch=-1)

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 32])

    best_acc = 0.0
    while epoch < args.n_epochs:
        audio_model.train()
        for i, (audio_input, audio_input_eng, audio_input_dur, phn_label, phns, utt_label, word_label, words) in enumerate(train_loader):
            audio_input = audio_input.to(device, non_blocking=True)
            audio_input_eng = audio_input_eng.to(device, non_blocking=True)
            audio_input_dur = audio_input_dur.to(device, non_blocking=True)
            audio_input = torch.cat([audio_input,audio_input_eng,audio_input_dur], dim=-1)
            phn_label = phn_label.to(device, non_blocking=True)
            utt_label = utt_label.to(device, non_blocking=True)
            word_label = word_label.to(device, non_blocking=True)

            # warmup
            warm_up_step = 100
            if global_step <= warm_up_step and global_step % 5 == 0:
                warm_lr = (global_step / warm_up_step) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            # add random noise for augmentation.
            noise = (torch.rand([audio_input.shape[0], audio_input.shape[1], audio_input.shape[2]]) - 1) * args.noise
            noise = noise.to(device, non_blocking=True)
            audio_input = audio_input + noise

            phn_loss_att, phn_acc, word_loss_att, word_acc, utt_ce, utt_acc = audio_model(audio_input, phns, words, word_label[:,:,-1], utt_label[:,0])

            loss = args.loss_w_phn * phn_loss_att + args.loss_w_word * word_loss_att + args.loss_w_utt * utt_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        print('Phone: loss: {:.3f}, acc: {:.3f}'.format(phn_loss_att, phn_acc))
        print('Word:, loss: {:.3f}, acc: {:.3f}'.format(word_loss_att, word_acc))
        print('Utt:, loss: {:.3f}, acc: {:.3f}'.format(utt_ce, utt_acc))
        print('Avg:, MSE: {:.3f}'.format(loss))
        
        writer.add_scalar("Loss/tr_phn", phn_loss_att, epoch)
        writer.add_scalar("Loss/tr_word", word_loss_att, epoch)
        writer.add_scalar("Loss/tr_utt", utt_ce, epoch)
        writer.add_scalar("Loss/total", loss, epoch)
        writer.add_scalar("Acc/tr_phn", phn_acc, epoch)
        writer.add_scalar("Acc/tr_word", word_acc, epoch)
        writer.add_scalar("Acc/tr_utt", utt_acc, epoch)

        acc_mean = np.mean([phn_acc, word_acc, utt_acc])
        if best_acc < acc_mean:
            print('Update best acc: acc_old: {:.3f}, acc_new: {:.3f}'.format(best_acc, acc_mean))
            best_acc = acc_mean
            best_epoch = epoch

        if best_epoch == epoch:
            if os.path.exists("%s/models/" % (exp_dir)) == False:
                os.mkdir("%s/models" % (exp_dir))
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))

        if global_step > warm_up_step:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        epoch += 1

class GoPDataset(Dataset):
    def __init__(self, set, am='librispeech'):
        # normalize the input to 0 mean and unit std.
        if am=='librispeech':
            dir='seq_data_librispeech_v3'
            norm_mean, norm_std = 3.203, 4.045
        elif am=='paiia':
            dir='seq_data_paiia'
            norm_mean, norm_std = -0.652, 9.737
        elif am=='paiib':
            dir='seq_data_paiib'
            norm_mean, norm_std = -0.516, 9.247
        else:
            raise ValueError('Acoustic Model Unrecognized.')

        if set == 'train':
            self.feat = torch.tensor(np.load('../data/'+dir+'/tr_feat.npy'), dtype=torch.float)
            self.feat_energy = torch.tensor(np.load('../data/'+dir+'/tr_energy_feat.npy'), dtype=torch.float)
            self.feat_dur = torch.tensor(np.load('../data/'+dir+'/tr_dur_feat.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load('../data/'+dir+'/tr_label_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load('../data/'+dir+'/tr_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load('../data/'+dir+'/tr_label_word.npy'), dtype=torch.float)
            self.word_id = torch.tensor(np.load('../data/'+dir+'/tr_word_id.npy'), dtype=torch.float)
        elif set == 'test':
            self.feat = torch.tensor(np.load('../data/'+dir+'/te_feat.npy'), dtype=torch.float)
            self.feat_energy = torch.tensor(np.load('../data/'+dir+'/te_energy_feat.npy'), dtype=torch.float)
            self.feat_dur = torch.tensor(np.load('../data/'+dir+'/te_dur_feat.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load('../data/'+dir+'/te_label_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load('../data/'+dir+'/te_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load('../data/'+dir+'/te_label_word.npy'), dtype=torch.float)
            self.word_id = torch.tensor(np.load('../data/'+dir+'/tr_word_id.npy'), dtype=torch.float)

        # normalize the GOP feature using the training set mean and std (only count the valid token features, exclude the padded tokens).
        self.feat = self.norm_valid(self.feat, norm_mean, norm_std)

        # normalize the utt_label to 0-2 (same with phn score range)
        self.utt_label = self.utt_label / 5
        # the last dim is word_id, so not normalizing
        self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5
        # phone related
        # [phn_id, phn_score]
        self.phn_label[:, :, 1] = self.phn_label[:, :, 1]

    # only normalize valid tokens, not padded token
    def norm_valid(self, feat, norm_mean, norm_std):
        norm_feat = torch.zeros_like(feat)
        for i in range(feat.shape[0]):
            for j in range(feat.shape[1]):
                if feat[i, j, 0] != 0:
                    norm_feat[i, j, :] = (feat[i, j, :] - norm_mean) / norm_std
                else:
                    break
        return norm_feat
 
    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, idx):
        # feat, phn_label, phn_id, utt_label, word_label
        #[word_id, phn_id]
        return self.feat[idx, :], self.feat_energy[idx, :], self.feat_dur[idx, :], self.phn_label[idx, :, 1], self.phn_label[idx, :, 0], self.utt_label[idx, :], self.word_label[idx, :], self.word_id[idx,:]


if __name__ == '__main__':
    args = parser.parse_args()

    am = args.am
    print('now train with {:s} acoustic models'.format(am))
    feat_dim = {'librispeech':84, 'paiia':86, 'paiib': 88}
    input_dim=feat_dim[am] + 7 + 1

    # nowa is the best models used in this work
    if args.model == 'gopt':
        print('now train a gopt_hierTFR_cls_ssl models')
        from models.gopt_hierAtt_v61_aspfix_pre import GOPT_hierAtt
        audio_mdl = GOPT_hierAtt(embed_dim=args.embed_dim, num_heads=args.goptheads, depth=args.goptdepth, input_dim=input_dim)
    # for ablation study only
    elif args.model == 'gopt_nophn':
        print('now train a GOPT models without canonical phone embedding')
        audio_mdl = GOPTNoPhn(embed_dim=args.embed_dim, num_heads=args.goptheads, depth=args.goptdepth, input_dim=input_dim)
    elif args.model == 'lstm':
        print('now train a baseline LSTM model')
        audio_mdl = BaselineLSTM(embed_dim=args.embed_dim, depth=args.goptdepth, input_dim=input_dim)

    tr_dataset = GoPDataset('train', am=am)
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    te_dataset = GoPDataset('test', am=am)
    te_dataloader = DataLoader(te_dataset, batch_size=2500, shuffle=False)

    train(audio_mdl, tr_dataloader, te_dataloader, args)
    writer.flush()
