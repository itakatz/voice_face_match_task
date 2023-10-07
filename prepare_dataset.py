''' load data files and prepare dataset for learning
'''
import os
import argparse
import pickle
import itertools
import pandas as pd
import numpy as np
from unidecode import unidecode 

def add_args(parser):
    parser.add_argument('--face-embed-file', type = str, required = True, help = 'path to file of face embeddings')
    parser.add_argument('--voice-embed-file', type = str, required = True, help = 'path to file of voice embeddings')
    parser.add_argument('--out-data-folder', type = str, default = 'data', help = 'path to folder where data files are saved')
    parser.add_argument('--random-seed', type = int, default = 42, help = 'random seed for sampling negative samples')
    
    return parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    if not os.path.isdir(args.out_data_folder):
        os.makedirs(args.out_data_folder)

    try:
        face_embed = pickle.load(open(args.face_embed_file, 'rb')) # pickle.load(open('vfm_assignment/image_embeddings.pickle', 'rb'))
        voice_embed = pickle.load(open(args.voice_embed_file, 'rb')) # pickle.load(open('vfm_assignment/audio_embeddings.pickle', 'rb'))
    except Exception as e:
        print(f'failed loading embeddings file(s): {e}')
        raise e

    #--- read features from dictionary to a list
    v_keys = []
    v_embeds = []
    for k, v in voice_embed.items():
        v_keys.append(k)
        v_embeds.append(v)
    
    f_keys = []
    f_embeds = []
    for k, v in face_embed.items():
        f_keys.append(k)
        f_embeds.append(v)

    # get names, treat the non-ascii issue (same person has different names in audio vs image folders)
    f_nms = pd.DataFrame([unidecode(k.split('/')[0]) for k in f_keys], columns = ['name'])
    v_nms = pd.DataFrame([unidecode(k.split('/')[0]) for k in v_keys], columns = ['name'])
    unq_ids = f_nms['name'].unique()
    
    v_embeds = np.r_[v_embeds]
    f_embeds = np.r_[f_embeds]

    #--- validation:0, test:1, train:2 (using Name's first letter, as in https://arxiv.org/pdf/1804.00326.pdf)
    v_nms['split'] = v_nms['name'].apply(lambda s: 0 if s[0] in ['A','B','C'] else (1 if s[0] in ['D','E'] else 2))
    f_nms['split'] = f_nms['name'].apply(lambda s: 0 if s[0] in ['A','B','C'] else (1 if s[0] in ['D','E'] else 2))

    #-- sampling: for each recording, take all combinations of pos voice with pos face, and add sampled negative face(s)
    for num_neg_per_pair in [1, 2, 5]:
        print(f'num negative samples per positive sample: {num_neg_per_pair}')
        np.random.seed(args.random_seed)
        #num_neg_per_pair = 5 #--- how many negative faces to samlpe for each pair of (v, f_pos)
        triplets = []
        for name in unq_ids:
            #--- first, construct the pair of voice and it's positive face(s)
            v_pos = v_nms[v_nms['name'] == name]
            f_pos = f_nms[f_nms['name'] == name]   
            pos_pairs = list(itertools.product(v_pos.index, f_pos.index))
            #--- sample negative face, make sure to not mix train/val/test
            split = v_pos['split'].iloc[0]
            f_neg = f_nms[(f_nms['name'] != name) & (f_nms['split'] == split)]
            
            #--- sample negative faces (TODO uniform sampling has bias to people with more images - probably negligible - ?)
            num_neg = len(pos_pairs) * num_neg_per_pair
            neg_inds = f_neg.sample(num_neg, replace = True).index #-- use replace = True since for the test set, the negative sample is not always big enough
            neg_inds = np.array(neg_inds).reshape((len(pos_pairs), num_neg_per_pair)) # more convenient indexing
            new_triplets = []
            for ipair, pair in enumerate(pos_pairs):
                new_triplets += [pair + (ind,) for ind in neg_inds[ipair]]
            triplets += new_triplets
            
        print(f'{len(triplets)} triplets created')  
        
        #--- split
        triplets = np.array(triplets)
        triplets_train = np.array([tr for tr in triplets if v_nms.loc[tr[0]].split == 2])
        triplets_val = np.array([tr for tr in triplets if v_nms.loc[tr[0]].split == 0])
        triplets_test = np.array([tr for tr in triplets if v_nms.loc[tr[0]].split == 1])

        #--- save to disk
        fnm_out = f'{args.out_data_folder}/data_num_neg_pp_{num_neg_per_pair}.pickle'
        pickle.dump([v_embeds, v_nms, f_embeds, f_nms, triplets, triplets_train, triplets_val, triplets_test], open(fnm_out, 'wb'))
        print(f'saved data to file: {fnm_out}')
    