import argparse
import pickle
import torch
import numpy as np

import voice_face_match as vfm
from voice_face_match_train import CFG # TODO cfg should be saved with the model, this is prone to bugs

def main(args):
    try:
        voice_embed, face1_embed, face2_embed, labels = pickle.load(open(args.input_test_file, 'rb'))
    except Exception as e:
        print(f'failed loading test data from file, error: {e}')
        return

    print(f'loaded {labels.shape[0]} test samples')
    
    voice_dim = 192
    face_dim = 512

    if len(labels.shape) == 1:
        labels = labels[:, np.newaxis]

    if voice_embed.shape[1] == voice_dim:
        pass
    elif voice_embed.shape[0] == voice_dim:
        voice_embed = voice_embed.T
    else:
        raise ValueError(f'voice embed should have dimenstion {voice_dim}')

    if face1_embed.shape[1] == face_dim:
        pass
    elif face1_embed.shape[0] == face_dim:
        face1_embed = face1_embed.T
        face2_embed = face1_embed.T
    else:
        raise ValueError(f'face embed should have dimenstion {face_dim}')

    #--- load model
    model = vfm.VoiceFaceTripletsClassifier(voice_dim, face_dim, CFG)
    model.load_state_dict(torch.load(args.model_file))

    #--- prepare data loader for the eval_model method
    use_cuda = torch.cuda.is_available()
    voice_embed, face1_embed, face2_embed, labels = [torch.Tensor(x.astype(np.float32)) for x in [voice_embed, face1_embed, face2_embed, labels]]
    if use_cuda:
        voice_embed, face1_embed, face2_embed, labels = [x.cuda() for x in [voice_embed, face1_embed, face2_embed, labels]]
        
    dataset = torch.utils.data.TensorDataset(voice_embed, face1_embed, face2_embed, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = False)  
    
    loss_fn = CFG['loss_fn']
    
    loss, accuracy = vfm.eval_model(model, loss_fn, dataloader, use_cuda)
    print(f'test loss: {loss:.3f} test accuracy: {accuracy:.3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-test-file', type = str, required = True, help = 'path to test data file (pickle file with embeddings')
    parser.add_argument('--model-file', type = str, required = True, help = 'path to model file to run')

    args = parser.parse_args()

    main(args)