# Prepare data for learning
run the script for data preparation, make sure path for embedding files (voice and image) are correct:
```
bash scripts/prepare_dataset.py
```

# Train model
run the script for training, optionally edit to set input file, results folder, and number of epochs
```
bash scripts/train.sh
```

Output example for a short test run (3 epochs):
```
input data file: data/data_num_neg_pp_2.pickle
saving results to folder: saved_models_TMP/num_neg_pp_2
loaded 142580 triplets
Feature batch shape: voice torch.Size([64, 192]), face torch.Size([64, 512])
Labels batch shape: torch.Size([64, 1])
[epoch 0, 5.68 sec] loss: train 0.697 val 0.693 accuracy: val 0.504 lr: 1.00e-03 
[epoch 1, 5.41 sec] loss: train 0.693 val 0.694 accuracy: val 0.505 lr: 9.95e-04 (*best validation loss*)
[epoch 2, 5.40 sec] loss: train 0.687 val 0.684 accuracy: val 0.551 lr: 9.90e-04 (*best validation loss*)
best validation-loss model saved to saved_models_TMP/num_neg_pp_2/best_model_epoch2.pth
final model saved to saved_models_TMP/num_neg_pp_2/final_model_epoch2.pth
final model (validation set): loss 0.684 acc 0.551
best model  (validation set): loss 0.684 acc 0.551
best model  (test set):       loss 0.683 acc 0.552
```
