from datasets import load_dataset
import tiktoken
import numpy as np
import os
from tqdm import tqdm


def process(example):
    ids = enc.encode_ordinary(example['text'])  # ignore special tokens
    ids.append(enc.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out


dataset = load_dataset("openwebtext")
# openwebtext only has train set, need to split test set
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')  # rename test set to val set

enc = tiktoken.get_encoding('gpt2')
num_proc = 1
tokenized = split_dataset.map(process, remove_columns=['text'], desc='tokenizing the splits', num_proc=num_proc)

for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.int64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len, ))
    total_batches = 1024
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx: idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()