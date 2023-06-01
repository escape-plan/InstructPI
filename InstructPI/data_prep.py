from datasets import Dataset
from datasets.dataset_dict import DatasetDict


class DatasetLoader:
    def __init__(self, train_df=None, dev_df=None, test_df=None, sample_size = 1):
        
        self.train_df = train_df.sample(frac = sample_size, random_state = 1999) if train_df is not None else train_df
        self.dev_df = dev_df
        self.test_df = test_df

    def reconstruct_strings(self, df, col):
        """
        Reconstruct strings to dictionaries when loading csv/xlsx files.
        """
        reconstructed_col = []
        for text in df[col]:
            if text != '[]' and isinstance(text, str):
                text = text.replace('[', '').replace(']', '').replace('{', '').replace('}', '').split(", '")
                req_list = []
                for idx, pair in enumerate(text):
                    splitter = ': ' if ': ' in pair else ':'
                    if idx%2==0:
                        reconstructed_dict = {} 
                        reconstructed_dict[pair.split(splitter)[0].replace("'", '')] = pair.split(splitter)[1].replace("'", '')
                    else:
                        reconstructed_dict[pair.split(splitter)[0].replace("'", '')] = pair.split(splitter)[1].replace("'", '')
                        req_list.append(reconstructed_dict)
            else:
                req_list = text
            reconstructed_col.append(req_list)
        df[col] = reconstructed_col
        return df

    def concat_with_truncate(self, text1, text2, bos_instruction, eos_instruction, max_len=510):
        max_input_len = max_len - len(bos_instruction) - len(eos_instruction)

        if len(text1)+len(text2) <= max_input_len:
            return bos_instruction.format(text1, text2)+ eos_instruction
        
        text1, text2 = text1.split(' '), text2.split(' ')
        
        longest = text1 if len(text1) > len(text2) else text2

        while sum(len(s) for s in text1) + sum(len(s) for s in text2) > max_input_len:
            longest.pop()
        
        text1, text2 = ' '.join(text1), ' '.join(text2)

        return bos_instruction.format(text1, text2)+ eos_instruction



    def create_data(self, df, bos_instruction = '', eos_instruction = '', max_len = 510):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        label_map = ['Positive', 'Negative']
        # label_map = ['No', 'Yes']
        # label_map = ['Non-paraphrase', 'Paraphrase']
        df['labels'] = df['labels'].apply(lambda x: label_map[0] if x == 0 else label_map[1])
        df['pair'] = df.apply(lambda row: bos_instruction.format(row['text1'], row['text2'])+ eos_instruction, axis=1)
        # df['pair'] = df.apply(lambda row: self.concat_with_truncate(row['text1'], row['text2'], bos_instruction, eos_instruction, max_len), axis=1)

        return df

    def set_data_for_training(self, tokenize_function):
        """
        Create the training and test dataset as huggingface datasets format.
        """
        # Define train and test sets
        dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df), 'dev': Dataset.from_pandas(self.dev_df), 'test': Dataset.from_pandas(self.test_df)})
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # if (self.train_df is not None) and (self.test_df is None):
        #     dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df)})
        #     tokenized_datasets = dataset.map(tokenize_function, batched=True)
        # elif(self.train_df is None) and (self.test_df is not None):
        #     dataset = DatasetDict({'test': Dataset.from_pandas(self.test_df)})
        #     tokenized_datasets = dataset.map(tokenize_function, batched=True)
        # elif (self.train_df is not None) and (self.test_df is not None):
        #     dataset = DatasetDict({'train': Dataset.from_pandas(self.train_df), 'test': Dataset.from_pandas(self.test_df)})
        #     tokenized_datasets = dataset.map(tokenize_function, batched=True)
        # else:
        #     dataset = {}
        #     tokenized_datasets = {}
        
        return dataset, tokenized_datasets
        