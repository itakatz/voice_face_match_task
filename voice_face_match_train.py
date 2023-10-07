import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
#import tqdm
import time
import os
import argparse

import voice_face_match as vfm

def main(data_file_name, results_folder, num_epochs):
    ''' Data files:
            data_num_neg_pp_1.pickle' - sample 1 negative face per positive sample
            data_num_neg_pp_2.pickle' - sample 2 negative faces per positive sample
            data_num_neg_pp_5.pickle' - sample 5 negative faces per positive sample
    '''
    #num_neg_per_pos_pair = 1 #--- must be 1,2, or 5
    #fnm = f'data/data_num_neg_pp_{num_neg_per_pos_pair}.pickle'
    print(f'input data file: {data_file_name}')
    print(f'saving results to folder: {results_folder}')
    
    try:
        v_embeds, v_nms, f_embeds, f_nms, triplets, triplets_train, triplets_val, triplets_test = pickle.load(open(data_file_name, 'rb'))
    except Exception as e:
        print(f'failed to load data from file {data_file_name}: {e}')
        return
        
    triplets = np.array(triplets)
    print(f'loaded {len(triplets)} triplets')
    
    #--- general params
    train_over_pairs = False
    batch_sz = 64
    #num_epochs = 100 # this is now input arg
    learning_rate = 0.001
    cfg = dict(input_layer_size = 256, dropout = 0.5)
    
    if train_over_pairs:
        random_switch_faces = False 
    else:
        random_switch_faces = True
        
    training_data = vfm.VoiceFaceDataset(v_embeds, f_embeds, triplets_train, random_switch_faces)
    validation_data = vfm.VoiceFaceDataset(v_embeds, f_embeds, triplets_val, random_switch_faces)
    test_data = vfm.VoiceFaceDataset(v_embeds, f_embeds, triplets_test, random_switch_faces)
    
    train_dataloader = DataLoader(training_data, batch_size = batch_sz, shuffle = True)
    val_dataloader = DataLoader(validation_data, batch_size = batch_sz, shuffle = False)
    test_dataloader = DataLoader(test_data, batch_size = batch_sz, shuffle = False)
    
    x_voice, x_face1, x_face2, train_labels = next(iter(train_dataloader))
    dims = [x_voice.shape[1], x_face1.shape[1]] # voice and face embeddings size
    print(f"Feature batch shape: voice {x_voice.size()}, face {x_face1.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    
    if train_over_pairs:
        VoiceFaceModel = vfm.VoiceFacePairsClassifier
    else:
        VoiceFaceModel = vfm.VoiceFaceTripletsClassifier
        
    model = VoiceFaceModel(dims[0], dims[1], cfg)
    loss_fn = nn.BCEWithLogitsLoss() # TODO impl a label smoothing class
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    sched = ExponentialLR(optimizer, gamma = 0.995, last_epoch=-1)

    #[ this is now given as input ] results_folder = f'saved_models/num_neg_pp_{num_neg_per_pos_pair}'
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
        
    use_cuda = torch.cuda.is_available()
    if use_cuda:
      print('using gpu')
      model.cuda()
    
    save_models = True
    model.train()
    stats = dict(train_loss = [], val_loss = [], val_acc = [])
    best_val_loss = np.inf
    last_saved = ''
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        t0 = time.time()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            x_voice, x_face1, x_face2, labels = data
            if use_cuda:
                x_voice = x_voice.cuda()
                x_face1 = x_face1.cuda()
                x_face2 = x_face2.cuda() 
                labels = labels.cuda()
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(x_voice, x_face1, x_face2)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
    
        t1 = time.time()
        
        #--- eval on val set and print stats 
        val_loss, val_acc = vfm.eval_model(model, loss_fn, val_dataloader, use_cuda)
        running_loss /= i
        stats['train_loss'].append(running_loss)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)
    
        is_best_str = ''
        if val_loss < best_val_loss and epoch > 0:
            best_val_loss = val_loss
            is_best_str = '(*best validation loss*)'
            if save_models:
                best_model_fnm = f'{results_folder}/best_model_epoch{epoch}.pth'
                torch.save(model.state_dict(), best_model_fnm)
                if os.path.exists(last_saved):
                    os.remove(last_saved)
                last_saved = best_model_fnm
                  
        print(f'[epoch {epoch}, {t1 - t0:.2f} sec] loss: train {running_loss:.3f} val {val_loss:.3f} accuracy: val {val_acc:.3f} lr: {sched.get_last_lr()[0]:.2e} {is_best_str}')
        sched.step()
    
    #--- also save last model
    if save_models:
        final_model_fnm = f'{results_folder}/final_model_epoch{epoch}.pth'
        torch.save(model.state_dict(), final_model_fnm)
        print(f'best validation-loss model saved to {best_model_fnm}')
        print(f'final model saved to {final_model_fnm}')
    
    best_fnm = best_model_fnm 
    final_fnm = final_model_fnm
    
    final_model = VoiceFaceModel(dims[0], dims[1], cfg)
    final_model.load_state_dict(torch.load(final_fnm))
    
    best_model = VoiceFaceModel(dims[0], dims[1], cfg)
    best_model.load_state_dict(torch.load(best_fnm))
    
    loss, acc = vfm.eval_model(final_model, loss_fn, val_dataloader, use_cuda)
    print(f'final model (validation set): loss {loss:.3f} acc {acc:.3f}')
    
    loss_best, acc_best = vfm.eval_model(best_model, loss_fn, val_dataloader, use_cuda)
    print(f'best model  (validation set): loss {loss_best:.3f} acc {acc_best:.3f}')
    
    test_loss_best, test_acc_best = vfm.eval_model(best_model, loss_fn, test_dataloader, use_cuda)
    print(f'best model  (test set):       loss {test_loss_best:.3f} acc {test_acc_best:.3f}')

    return best_fnm, final_fnm, stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data-file', type = str, required = True, help = 'path to training data file (as prepared with "scripts/prepare_dataset.sh")')
    parser.add_argument('--results-folder', type = str, required = True, help = 'path to folder for saving result models')
    parser.add_argument('--epochs', type = int, default = 100, help = 'number of epochs to run training')
    
    args = parser.parse_args()
    
    best_fnm, final_fnm, stats = main(args.input_data_file, args.results_folder, args.epochs)