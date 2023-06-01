from InstructPI.utils import T5Generator
import pandas as pd
from instructions import InstructionsHandler
from InstructPI.data_prep import DatasetLoader

model_out_path = '/public/home/hongy/ray/InstructPI/Models/flan-t5-large-QQP/checkpoint-30321'
t5_exp = T5Generator(model_out_path)
experiment_name = 'QQP'

train_file_path = '../dataset/{}/dev'.format(experiment_name)
dev_file_path = '../dataset/{}/dev'.format(experiment_name)

header_name = ['text1', 'text2', 'labels']
tr_df = pd.read_csv(train_file_path, sep = '\t', header=None, names = header_name)
dev_df = pd.read_csv(dev_file_path, sep = '\t', header=None, names = header_name)

if experiment_name == 'QQP':
    test_file_path = dev_file_path
else:
    test_file_path = '../dataset/{}/test'.format(experiment_name)

test_df = pd.read_csv(test_file_path, sep = '\t', header=None, names = header_name)

instruct_handler = InstructionsHandler()
instruct_handler.load_instruction_set()

loader = DatasetLoader(tr_df, dev_df, test_df)

if loader.train_df is not None:
    loader.train_df = loader.create_data(loader.train_df, instruct_handler.pi['bos_instruct'], instruct_handler.pi['eos_instruct'])

if loader.dev_df is not None:
    loader.dev_df = loader.create_data(loader.dev_df, instruct_handler.pi['bos_instruct'], instruct_handler.pi['eos_instruct'])
if loader.test_df is not None:
    loader.test_df = loader.create_data(loader.test_df, instruct_handler.pi['bos_instruct'], instruct_handler.pi['eos_instruct'])

ds, tokenized_ds = loader.set_data_for_training(t5_exp.tokenize_function_inputs)

# Get prediction labels - Testing set
te_pred_labels = t5_exp.get_labels(tokenized_dataset = tokenized_ds, sample_set = 'test', batch_size = 64)
te_labels = [i.strip() for i in ds['test']['labels']]

unexpected_ans = []
for i in unexpected_ans:
    if i.lower() not in ['yes', 'no', 'paraphrase', 'non-paraphrase']:
        unexpected_ans.append(i)
print(te_pred_labels[:10])
print('unexpected_ans: ', unexpected_ans)

acc = t5_exp.get_metrics(te_labels, te_pred_labels)
print('Test acc: ', acc)