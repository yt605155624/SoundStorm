# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/t2s_dataset.py
'''
1. 剔除不是 speech 类型的样本
2. 剔除过长的样本
3. 剔除 phoneme/sec 太大或太小的样本, fengyufei's 上下限 8-30
   直方图 https://github.com/yt605155624/mine_images/issues/1#issuecomment-1683696942, 6 ~ 22 比较合适
'''
import random
import time
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import torch
from soundstorm.s1.AR.data.dataset import batch_sequences
from soundstorm.s1.AR.text_processing.phonemizer import GruutPhonemizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def get_key_name(file_path: str):
    name_list = file_path.split("/")
    # phonemes_0_3.npy -> 0_3
    rank_name = '_'.join(name_list[-1].split('.')[0].split('_')[-2:])
    # small/medium/large/duplicate_0_3
    key_name = f'{name_list[-3]}_{rank_name}'
    return key_name


class Text2SemanticDataset(Dataset):
    """dataset class for text tokens to semantic model training."""

    def __init__(
            self,
            phoneme_paths,
            semantic_paths,
            non_speech_paths=None,
            max_sample=None,
            max_sec: int=100,
            pad_val: int=1024,
            # min value of phoneme/sec 
            min_ps_ratio: int=6,
            # max value of phoneme/sec 
            max_ps_ratio: int=22, ) -> None:
        super().__init__()

        self.semantic_data_dict = dict()
        self.phoneme_data_dict = dict()
        self.non_speech_data_dict = dict()

        semantic_files = semantic_paths
        phoneme_files = phoneme_paths
        non_speech_files = non_speech_paths

        print("semantic_files:", semantic_files)
        print("phoneme_files:", phoneme_files)
        print("non_speech_files:", non_speech_files)

        s_st = time.time()
        for semantic_file in semantic_files:
            key_name = get_key_name(semantic_file)
            self.semantic_data_dict[key_name] = pd.read_csv(
                semantic_file, delimiter='\t')
        print(f"load semantic_files done, cost {round(time.time()-s_st, 2)}s")
        p_st = time.time()
        for phoneme_file in phoneme_files:
            key_name = get_key_name(phoneme_file)
            self.phoneme_data_dict[key_name] = np.load(
                phoneme_file, allow_pickle=True).item()
        print(f"load phoneme_files done, cost {round(time.time()-p_st, 2)}s")
        n_st = time.time()
        for non_speech_file in non_speech_files:
            key_name = get_key_name(non_speech_file)
            self.non_speech_data_dict[key_name] = np.load(
                non_speech_file, allow_pickle=True).item()
        print(f"load non_speech_files done, cost {round(time.time()-n_st, 2)}s")

        # pad for semantic tokens
        self.PAD: int = pad_val
        self.hz = 50
        # max seconds of semantic token
        self.max_sec = max_sec
        self.min_ps_ratio = min_ps_ratio
        self.max_ps_ratio = max_ps_ratio
        self.phonemizer: GruutPhonemizer = GruutPhonemizer(language='en-us')
        # max sample per split (key_name)
        self.max_sample = max_sample

        # {idx: (semantic, phoneme)}
        # semantic list, phoneme list
        self.semantic_phoneme = {}
        self.item_names = []

        self.inited = False
        self.total_semantic_data_len = 0
        self.total_phoneme_data_len = 0

        self.idx = 0
        self.num_not_in = 0
        self.num_deleted_bigger = 0
        self.num_deleted_ps = 0
        self.num_deleted_none_speech = 0
        print("self.semantic_data_dict.keys():", self.semantic_data_dict.keys())
        # 一定要转成 list 否则会报错 RuntimeError: dictionary changed size during iteration
        # 要么把 L117 for 换成 while
        key_names = list(self.semantic_data_dict.keys())
        if not self.inited:
            # 调用初始化函数
            for key_name in key_names:
                i_st = time.time()
                self.init_batch(key_name)
                print(
                    f"init_batch of {key_name} done, cost {round(time.time()-i_st, 2)}s"
                )
            self.inited = True
            print("self.total_semantic_data_len:", self.total_semantic_data_len)
            print("self.total_phoneme_data_len:", self.total_phoneme_data_len)

            if self.num_deleted_none_speech > 0:
                print(
                    f"deleted {self.num_deleted_none_speech} audios who's audio tag are not 'speech'"
                )

            if self.num_not_in > 0:
                print(
                    f"there are {self.num_not_in} semantic datas not in phoneme datas"
                )

            if self.num_deleted_bigger > 0:
                # 300 for small, 3461 for small + medium
                print(
                    f"deleted {self.num_deleted_bigger} audios who's duration are bigger than {self.max_sec} seconds"
                )
            if self.num_deleted_ps > 0:
                print(
                    f"deleted {self.num_deleted_ps} audios who's phoneme/sec are bigger than {self.max_ps_ratio} or smaller than {self.min_ps_ratio}"
                )
            # 117876 for small, 1189168 for small + medium
            print("dataset.__len__():", self.__len__())

    def init_batch(self, key_name: str):
        if key_name not in self.phoneme_data_dict.keys():
            print(f'{key_name} not in self.phoneme_data_dict')
            return None

        semantic_data = self.semantic_data_dict[key_name]
        phoneme_data = self.phoneme_data_dict[key_name]
        non_speech_data = None

        if key_name in self.non_speech_data_dict:
            non_speech_data = self.non_speech_data_dict[key_name]

        if self.max_sample is not None:
            semantic_data = semantic_data[:self.max_sample]

        semantic_data_len = len(semantic_data)
        phoneme_data_len = len(phoneme_data.keys())

        self.total_semantic_data_len += semantic_data_len
        self.total_phoneme_data_len += phoneme_data_len

        for i in range(semantic_data_len):
            # 先依次遍历
            # get str
            item_name = semantic_data['item_name'][i]
            # 过滤掉非 speech 类型的样本

            if non_speech_data is not None and item_name in non_speech_data.keys(
            ):
                self.num_deleted_none_speech += 1
                continue
            # ASR 结果存在为空的情况 (txt_*.npy 里就没有这个 key)
            # or ASR 结果不为空但是转换为 phoneme 时报错
            # 都会导致 phoneme_*.npy 中没有这个 key
            try:
                phoneme = phoneme_data[item_name]
            except Exception:
                # print(f"{item_name} not in phoneme_data !")
                self.num_not_in += 1
                continue

            semantic_str = semantic_data['semantic_audio'][i]
            # get token list
            semantic_ids = [int(idx) for idx in semantic_str.split(' ')]
            # (T), 是否需要变成 (1, T) -> 不需要，因为需要求 len
            # 过滤掉太长的样本
            if len(semantic_ids) > self.max_sec * self.hz:
                self.num_deleted_bigger += 1
                continue

            # (T, ), 这个速度不会很慢，所以可以在一开始就处理，无需在 __getitem__ 里面单个处理
            phoneme_ids = self.phonemizer.transform(phoneme)
            # 过滤掉 phoneme / sec 的极端值
            ps_ratio = len(phoneme_ids) / (len(semantic_ids) / self.hz)
            if ps_ratio > self.max_ps_ratio or ps_ratio < self.min_ps_ratio:
                self.num_deleted_ps += 1
                continue

            self.semantic_phoneme[self.idx] = (semantic_ids, phoneme_ids)
            self.idx += 1
            self.item_names.append(item_name)
            # 垃圾回收否则可能内存不够
            del semantic_str
            del phoneme
        del self.semantic_data_dict[key_name]
        del self.phoneme_data_dict[key_name]

        if key_name in self.non_speech_data_dict:
            del self.non_speech_data_dict[key_name]

    def __get_item_names__(self) -> List[str]:
        return self.item_names

    def __len__(self) -> int:
        return len(self.semantic_phoneme.keys())

    def __getitem__(self, idx: int) -> Dict:
        semantic_ids, phoneme_ids = self.semantic_phoneme[idx]
        phoneme_ids_len = len(phoneme_ids)
        # semantic tokens target
        semantic_ids_len = len(semantic_ids)
        return {
            'idx': idx,
            'phoneme_ids': phoneme_ids,
            'phoneme_ids_len': phoneme_ids_len,
            'semantic_ids': semantic_ids,
            'semantic_ids_len': semantic_ids_len
        }

    def get_sample_length(self, idx: int):
        semantic_ids = self.semantic_phoneme[idx][0]
        sec = 1.0 * len(semantic_ids) / self.hz
        return sec

    def collate(self, examples: List[Dict]) -> Dict:
        sample_index: List[int] = []
        phoneme_ids: List[torch.Tensor] = []
        phoneme_ids_lens: List[int] = []
        semantic_ids: List[torch.Tensor] = []
        semantic_ids_lens: List[int] = []

        for item in examples:
            sample_index.append(item["idx"])
            phoneme_ids.append(np.array(item["phoneme_ids"], dtype=np.int64))
            semantic_ids.append(np.array(item["semantic_ids"], dtype=np.int64))
            phoneme_ids_lens.append(item["phoneme_ids_len"])
            semantic_ids_lens.append(item["semantic_ids_len"])

        # pad 0
        phoneme_ids = batch_sequences(phoneme_ids)
        semantic_ids = batch_sequences(semantic_ids, pad_value=self.PAD)

        # # convert each batch to torch.tensor
        phoneme_ids = torch.tensor(phoneme_ids)
        semantic_ids = torch.tensor(semantic_ids)
        phoneme_ids_lens = torch.tensor(phoneme_ids_lens)
        semantic_ids_lens = torch.tensor(semantic_ids_lens)

        return {
            # List[int]
            "ids": sample_index,
            # torch.Tensor (B, max_phoneme_length) 
            "phoneme_ids": phoneme_ids,
            # torch.Tensor (B)
            "phoneme_ids_len": phoneme_ids_lens,
            # torch.Tensor (B, max_semantic_ids_length)
            "semantic_ids": semantic_ids,
            # torch.Tensor (B)
            "semantic_ids_len": semantic_ids_lens,
        }


class DDPSyncSampler(object):
    def __init__(self,
                 size,
                 seed,
                 global_rank,
                 local_rank,
                 world_size,
                 shuffle=True):
        # DataSet 中总 batch 数，
        self.size = size
        self.seed = seed
        # global_rank, 对完整数据集也是按照 global_rank 数划分的
        self.rank = global_rank
        self.epoch = 0
        self.shuffle = shuffle
        # ❗️ Ensure all GPUs have the same number of batches by adding redundant
        # sampling indexes. The redundant samples are not shuffled.
        # Cannot access the "local_rank" variable so it is a bit dirty
        # get local rank
        # 此处 local_rank 主要是为了设置 device
        device = torch.device(f"cuda:{local_rank}")
        size = torch.Tensor([size]).to(device)
        # 获得总卡数
        # dist.get_world_size 对应 dist.get_rank() 是 global_rank
        # 为 global_rank 的每个 rank 都造一个 size
        gathered_size = [size for _ in range(world_size)]
        # list 里面每个值依旧一样
        torch.distributed.all_gather(gathered_size, size)
        # 计算当前 gpu 需要 padding 的 batch 数量
        # always 0
        self.pad_number = int(max(gathered_size).item() - size.item())
        # 对 batch index 进行一系列操作, 但是长度还是与 dataset.__len__() 保持一致
        self.refresh()

    def refresh(self):
        seq = list(range(self.size))
        # introduce local randomness by local random shuffling
        # otherwise each global batch will be identical across epochs
        chunk_size, start = 10, 0
        random.seed(self.rank + self.seed + self.epoch)
        while start < self.size:
            # 分段
            seg = seq[start:min(self.size, start + chunk_size)]
            # 打乱该段
            local_random_order = random.sample(list(range(len(seg))), len(seg))
            seg = [seg[i] for i in local_random_order]
            seq[start:min(self.size, start + chunk_size)] = seg
            start += len(seg)

        # even after this shuffle, the batch lengths across GPUs 
        # are very similar
        # 上面是段内打乱，这里是段和段的顺序打乱
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(seq)

        if self.pad_number > 0:
            seq = list(range(self.pad_number)) + seq
        # list(int), 每个值表示 batch index
        self.seq = seq
        self.epoch += 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)

    def get_state_dict(self):
        state_dict = {
            'epoch': self.epoch,
            'seed': self.seed,
        }
        return state_dict

    def load_state_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)


if __name__ == '__main__':
    root_dir_1 = '/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/ar_s1/SoundStorm/dump_librilight/small/train/'
    root_dir_2 = '/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/ar_s1/SoundStorm/dump_librilight/medium/train/'
    start_build_time = time.time()
    dataset = Text2SemanticDataset(
        phoneme_dirs=[root_dir_1, root_dir_2],
        semantic_dirs=[root_dir_1, root_dir_2],
        non_speech_dirs=[root_dir_1])
    batch_size = 12
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate,
        shuffle=False)
    # small + medium dev: 3.7s, 21189 seqs
    # small + medium train: 300s, 1192629 seqs
    print(f"time of build dataloader: {time.time() - start_build_time}")
    # for i, batch in enumerate(dataloader):
    #     if i == 0:
    #         print('batch["ids"]:', batch["ids"])
    #         print('batch["phoneme_ids"]:', batch["phoneme_ids"],
    #               batch["phoneme_ids"].shape)
    #         print('batch["phoneme_ids_len"]:', batch["phoneme_ids_len"],
    #               batch["phoneme_ids_len"].shape)
    #         print('batch["semantic_ids"]:', batch["semantic_ids"],
    #               batch["semantic_ids"].shape)
    #         print('batch["semantic_ids_len"]:', batch["semantic_ids_len"],
    #               batch["semantic_ids_len"].shape)
