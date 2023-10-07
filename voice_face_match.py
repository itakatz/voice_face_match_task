import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

#--- Dataset class
class VoiceFaceDataset(Dataset):
    def __init__(self, v_embeds, f_embeds, triplets, random_switch_faces = False, random_seed = 42):        
        #self.v_nms = v_nms
        #self.v_nms = f_nms
        self.v_embeds = v_embeds.astype(np.float32)
        self.f_embeds = f_embeds.astype(np.float32)
    
        self.triplets = triplets.copy() #--- we change it in-place so make a copy
        N = len(self.triplets)
        
        y = np.ones((N, 1)) #.astype(int) #--- we begin with all samples having the pos-face as the 1st face-sample
        #--- random shuffle of pos/neg faces
        if random_switch_faces:
            np.random.seed(random_seed)
            fpos_neg = self.triplets[:, 1:]
            
            i_switch = np.random.choice(N, N // 2, replace = False)
            fpos_neg[i_switch] = fpos_neg[i_switch, ::-1]
            self.triplets[:, 1:] = fpos_neg #--- the randomly shuffled columns            
            y[i_switch] = 0.
            
        self.labels = y

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        #--- to return range of samples:
        #idx = [idx] if type(idx) is int else idx
        #X = np.r_[self.v_embeds[vpos, :], self.f_embeds[f1, :], self.f_embeds[f2, :]]
        
        triplets = self.triplets[idx]
        vpos, f1, f2 = triplets.T
                
        v = self.v_embeds[vpos, :]
        f1 = self.f_embeds[f1, :]
        f2 = self.f_embeds[f2, :]
        #--- returning a triplet without concatenation it is more convenient/flexibil later in model 
        #X = np.r_[v, f1, f2]
        labels = self.labels[idx]
        
        return v, f1, f2, labels

class VoiceFaceTripletsClassifier(nn.Module):
    ''' model that classifies triplets of embeddings by concatenation [voice, face_pos, face_neg]
    '''
    def __init__(self, input_sz_voice, input_sz_face, cfg):
        super().__init__()
        input_layer_size = cfg['input_layer_size']
        dropout = cfg['dropout']
        dim = input_sz_voice + 2 * input_sz_face
        
        self.fc1 = nn.Linear(dim, input_layer_size)
        self.fc2 = nn.Linear(input_layer_size // 2, input_layer_size // 4)
        self.fc3 = nn.Linear(input_layer_size // 8, 1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        #self.dropout2 = nn.Dropout(0.1)
        self.batch_norm1 = nn.BatchNorm1d(num_features = input_layer_size // 2)
        #self.batch_norm2 = nn.BatchNorm1d(num_features = input_layer_size // 8)

    def forward(self, x_v, x_f1, x_f2):
        ''' x_v  - features of pos voice
            x_f1 - features of 1st face
            x_f2 - features of 2nd face
        '''
        x = torch.cat((x_v, x_f1, x_f2), dim=1)
        x = self.pool(F.relu(self.fc1(x)))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = self.pool(F.relu(self.fc2(x)))
        #x = self.dropout2(x)
        #x = self.batch_norm2(x)
        x = self.fc3(x)
        
        return x

class VoiceFacePairsClassifier(nn.Module):
    ''' model that classifies pairs of embeddings by concatenation [voice, face_pos] and [voice, face_neg]
    '''
    def __init__(self, input_sz_voice, input_sz_face, cfg):
        super().__init__()
        input_layer_size = cfg['input_layer_size']
        dropout = cfg['dropout']        
        dim = input_sz_voice + input_sz_face
        
        self.fc1 = nn.Linear(dim, input_layer_size)
        self.fc2 = nn.Linear(input_layer_size // 2, input_layer_size // 4)
        self.fc3 = nn.Linear(input_layer_size // 8, 1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        #self.dropout2 = nn.Dropout(0.1)
        self.batch_norm1 = nn.BatchNorm1d(num_features = input_layer_size // 2)
        #self.batch_norm2 = nn.BatchNorm1d(num_features = input_layer_size // 8)

    def forward(self, x_v, x_f1, x_f2):
        ''' x_v  - features of pos voice
            x_f1 - features of 1st face
            x_f2 - features of 2nd face
        '''
        output = []
        for x_f in [x_f1, x_f2]:
            x = torch.cat((x_v, x_f), dim=1)
            x = self.pool(F.relu(self.fc1(x)))
            x = self.batch_norm1(x)
            x = self.dropout(x)
            x = self.pool(F.relu(self.fc2(x)))
            #x = self.dropout2(x)
            #x = self.batch_norm2(x)
            x = self.fc3(x)
            output.append(x)

        #--- the score for the triplet is [score of pos pair] - [score of neg pair]
        #--- this behaves as a normal logit which is > 0 for predicting "1" and < 0 for predicting "0"
        return output[0] - output[1]

def eval_model(model, loss_fn, dataloader, use_cuda):
    is_train = model.training
    if is_train:
        model.eval()
    
    with torch.no_grad():
        if use_cuda:
            preds_lbls = [(model(x.cuda(), f1.cuda(), f2.cuda()), y.cuda()) for x, f1, f2, y in dataloader]
        else:
            preds_lbls = [(model(x, f1, f2), y) for x, f1, f2, y in dataloader]

        loss = [loss_fn(pred, y).item() for pred, y in preds_lbls]             
        loss = np.mean(loss)
        
        acc = [((pred.detach().cpu().numpy() >= 0.) == y.cpu().numpy())[:, 0] for pred, y in preds_lbls]
            
        acc = np.concatenate(acc).mean()

    if is_train:
        model.train()
        
    return loss, acc