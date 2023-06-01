import os
import torch
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from InstructPI.data_prep import DatasetLoader
from InstructPI.utils import T5Generator
from instructions import InstructionsHandler

use_mps = True if torch.has_mps else False
root_path = './'
os.chdir(root_path)

experiment_name = 'QQP'
model_checkpoint = '/public/home/hongy/pretrained_models/flan-t5-base'
print('Experiment Name: ', experiment_name)
model_out_path = './Models'
model_out_path = os.path.join(model_out_path, f"{model_checkpoint.split('/')[-1]}-{experiment_name}")
print('Model output path: ', model_out_path)

# Load the data
train_file_path = '../dataset/{}/train'.format(experiment_name)
dev_file_path = '../dataset/{}/dev'.format(experiment_name)

header_name = ['text1', 'text2', 'labels']
tr_df = pd.read_csv(train_file_path, sep = '\t', header=None, names = header_name)
dev_df = pd.read_csv(dev_file_path, sep = '\t', header=None, names = header_name)

if experiment_name == 'QQP':
    test_file_path = dev_file_path
else:
    test_file_path = '../dataset/{}/test'.format(experiment_name)

te_df = pd.read_csv(test_file_path, sep = '\t', header=None, names = header_name)

instruct_handler = InstructionsHandler()
instruct_handler.load_instruction_set()

loader = DatasetLoader(tr_df, dev_df, te_df)

if loader.train_df is not None:
    loader.train_df = loader.create_data(loader.train_df, instruct_handler.pi['bos_instruct'], instruct_handler.pi['eos_instruct'])
if loader.dev_df is not None:
    loader.dev_df = loader.create_data(loader.dev_df, instruct_handler.pi['bos_instruct'], instruct_handler.pi['eos_instruct'])
if loader.test_df is not None:
    loader.test_df = loader.create_data(loader.test_df, instruct_handler.pi['bos_instruct'], instruct_handler.pi['eos_instruct'])
    
t5_exp = T5Generator(model_checkpoint)


ds, tokenized_ds = loader.set_data_for_training(t5_exp.tokenize_function_inputs)

training_args = {
    'output_dir':model_out_path,
    'evaluation_strategy':"steps",
    'learning_rate':5e-5,
    'lr_scheduler_type':'cosine',
    'per_device_train_batch_size':32,
    'per_device_eval_batch_size':10,
    'num_train_epochs':5,
    'weight_decay':0.01,
    'warmup_ratio':0.1,
    'save_strategy':'epoch',
    'load_best_model_at_end':False,
    'push_to_hub':False,
    'eval_steps': 2000,
    'eval_accumulation_steps':1,
    'predict_with_generate':True,
    'use_mps_device':use_mps
}

model_trainer = t5_exp.train(tokenized_ds, **training_args)