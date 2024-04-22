# This file adds the ability to embed a 4th language to the pretrained model

import torch
import pathlib

#if linux
#pathlib.WindowsPath = pathlib.PosixPath
device = torch.device("cpu")
checkpoint = torch.load('checkpoints/vallex-checkpoint.pt')

checkpoint['model']['ar_language_embedding.word_embeddings.weight'] = torch.cat(
    [checkpoint['model']['ar_language_embedding.word_embeddings.weight'].to(device), torch.rand(1, 1024).to(device)])
checkpoint['model']['nar_language_embedding.word_embeddings.weight'] = torch.cat(
    [checkpoint['model']['nar_language_embedding.word_embeddings.weight'].to(device), torch.rand(1, 1024).to(device)])

torch.save(checkpoint, 'vallex-checkpoint.pt')
