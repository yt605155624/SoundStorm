"""
1. read text of dataset, for LibriLight read txt_*.npy -> 需要整理成 list(utt_id, txt) 的形式
2. text -> IPA by GruutPhonemizer
3. save out a *.npy dict for all text
4. LibriLight 每个 split 分开处理
my_dict = {"utt_id1": text1, "utt_id2": text2}
np.save(output_filename, my_dict)
my_dict = np.load(output_filename, allow_pickle=True).item()
"""
import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
from pathlib import Path

import numpy as np
import tqdm
from soundstorm.s1.AR.text_processing.phonemizer import GruutPhonemizer
from soundstorm.s2.exps.hubert.feature_utils import get_shard_range
from soundstorm.utils import check_txt_file


def process_sentence(item, phonemizer, output_dir):
    utt_id, text = item
    phonemes_dir = output_dir / "phonemes"
    phonemes_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(phonemes_dir, 0o777)
    phonemes_path = phonemes_dir / (utt_id + ".txt")
    try:
        if os.path.exists(phonemes_path) and check_txt_file(phonemes_path):
            # print(phonemes_path, 'exits!')
            pass
        else:
            phonemes = phonemizer.phonemize(text, espeak=False)
            with open(phonemes_path, 'w') as f:
                f.write(phonemes)
            # -rw-r--r-- => rw-rw-rw-
            os.chmod(phonemes_path, 0o666)
        record = {"utt_id": utt_id, "phonemes_path": phonemes_path}
    except Exception:
        print("occur Exception")
        traceback.print_exc()
        return None
    return record


def process_sentences(args, items, phonemizer, output_dir, nprocs: int=1):
    print("nprocs:", nprocs)
    if nprocs == 1:
        results = []
        for item in tqdm.tqdm(items, total=len(items)):
            record = process_sentence(
                item=item, phonemizer=phonemizer, output_dir=output_dir)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(items)) as progress:
                for item in items:
                    future = pool.submit(process_sentence, item, phonemizer,
                                         output_dir)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)

    results.sort(key=itemgetter("utt_id"))

    npy_dict = {}
    print(f"start to save {args.rank}_{args.nshard}.npy ...")
    save_start_time = time.time()
    for item in tqdm.tqdm(results, total=len(results), colour='green'):
        # 这里加 try, 因为 txt 文件可能损坏
        try:
            utt_id = item["utt_id"]
            phonemes = check_txt_file(item["phonemes_path"])
            if phonemes is not False:
                npy_dict[utt_id] = phonemes
            else:
                print(f'phonemes of {utt_id} is False')
        except Exception:
            print(f"{utt_id} occur Exception")
            traceback.print_exc()
            continue

    filename = output_dir / f'phonemes_{args.rank}_{args.nshard}.npy'
    np.save(filename, npy_dict)
    print(f"npy file '{filename}' write down")
    print(f"time of save stage: {round(time.time() - save_start_time,2)}s")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Get phones for 11labs datasets")

    parser.add_argument(
        "--data_dir", default=None, type=str, help="directory to dataset.")

    parser.add_argument(
        "--dump_dir",
        type=str,
        required=True,
        help="directory to dump feature files.")
    parser.add_argument(
        "--num-cpu", type=int, default=1, help="number of process.")

    parser.add_argument("--nshard", type=int, default=3)
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()
    print(f"nshard: {args.nshard}, rank: {args.rank}")

    data_dir = Path(args.data_dir).expanduser()
    dump_dir = Path(args.dump_dir).expanduser()
    # use absolute path
    dump_dir = dump_dir.resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)
    # drwxr-xr-x => rwxrwxrwx
    os.chmod(dump_dir, 0o777)

    assert data_dir.is_dir()

    # olny spk_id in list, sort by lexicographical order 
    speaker_list = sorted(os.listdir(data_dir))
    print("len(speaker_list):", len(speaker_list))
    start, end = get_shard_range(len(speaker_list), args.nshard, args.rank)
    # speaker_list for this rank
    speaker_list = speaker_list[start:end]

    train_txts = []
    dev_txts = []
    test_txts = []
    sub_num_dev = 10

    for speaker in speaker_list:
        data_dict = {}
        text_path = data_dir / speaker / 'metadata-merge.csv'
        with open(text_path, 'r') as rf:
            for line in rf:
                line_list = line.strip().split('|')
                utt_id = line_list[0]
                raw_text = line_list[-1]
                data_dict[utt_id] = raw_text

        # sort 之后再分 train / dev / test 保证和音频的划分一致
        sorted_dict = sorted(data_dict.items())

        if len(sorted_dict) > 100:
            train_txts += sorted_dict[:-sub_num_dev * 2]
            dev_txts += sorted_dict[-sub_num_dev * 2:-sub_num_dev]
            test_txts += sorted_dict[-sub_num_dev:]
        else:
            train_txts += sorted_dict

    print(
        f"num of txt files in rank {args.rank} / {args.nshard}: {len(train_txts)+len(dev_txts) + len(test_txts)}"
    )

    train_dump_dir = dump_dir / "train"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(train_dump_dir, 0o777)
    dev_dump_dir = dump_dir / "dev"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(dev_dump_dir, 0o777)
    test_dump_dir = dump_dir / "test"
    test_dump_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(test_dump_dir, 0o777)

    phonemizer = GruutPhonemizer(language='en-us')

    # process for the 3 sections
    if train_txts:
        process_sentences(
            args=args,
            items=train_txts,
            output_dir=train_dump_dir,
            phonemizer=phonemizer,
            nprocs=args.num_cpu)
    if dev_txts:
        process_sentences(
            args=args,
            items=dev_txts,
            output_dir=dev_dump_dir,
            phonemizer=phonemizer,
            nprocs=args.num_cpu)
    if test_txts:
        process_sentences(
            args=args,
            items=test_txts,
            output_dir=test_dump_dir,
            phonemizer=phonemizer,
            nprocs=args.num_cpu)


if __name__ == "__main__":
    main()
