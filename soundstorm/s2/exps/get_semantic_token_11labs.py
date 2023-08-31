import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch
import tqdm
from soundstorm.s2.exps.hubert.feature_utils import get_shard_range
from soundstorm.s2.models.hubert.semantic_tokenizer import SemanticTokenizer
from soundstorm.utils import check_numpy_file


def process_sentence(args, fp: Path, output_dir: Path, semantic_tokenizer):
    utt_id = fp.stem
    sr = args.sr
    record = None
    semantic_token_dir = output_dir / "semantic_token"
    semantic_token_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(semantic_token_dir, 0o777)
    try:
        semantic_token_path = semantic_token_dir / (utt_id + ".npy")
        if os.path.exists(semantic_token_path) and check_numpy_file(
                semantic_token_path):
            # print(semantic_token_path, 'exits!')
            pass
        else:
            # reading, resampling may occur
            wav, _ = librosa.load(str(fp), sr=sr)
            wav = torch.tensor(wav).unsqueeze(0)
            semantic_token = semantic_tokenizer.tokenize(wav)
            semantic_token_np = semantic_token.detach().cpu().numpy()
            np.save(semantic_token_path, semantic_token_np)
            # -rw-r--r-- => rw-rw-rw-
            os.chmod(semantic_token_path, 0o666)
        record = {"utt_id": utt_id, "semantic_token_path": semantic_token_path}
    except Exception:
        print("occur Exception")
        traceback.print_exc()
        return None
    return record


def process_sentences(args,
                      fps: Path,
                      output_dir: Path,
                      semantic_tokenizer,
                      nprocs: int=1):
    if nprocs == 1:
        results = []
        for fp in tqdm.tqdm(fps, total=len(fps)):
            record = process_sentence(
                args=args,
                fp=fp,
                output_dir=output_dir,
                semantic_tokenizer=semantic_tokenizer)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp in fps:
                    future = pool.submit(process_sentence, args, fp, output_dir,
                                         semantic_tokenizer)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)

    data = [['item_name', 'semantic_audio']]
    print(f"start to save {args.rank}_{args.nshard}.tsv ...")
    save_start_time = time.time()
    # record 是 List of Dict, 一条大 wav 一个 record，一条小 wav 一个 sub_recored
    for record in tqdm.tqdm(results, total=len(results), colour='green'):
        try:
            utt_id = record["utt_id"]
            # old hubert_kmeans shape is (T,), new hubert_kmeans shape is (1, T)
            # so add [0] here
            semantic_token = np.load(record["semantic_token_path"])[0].tolist()
            semantic_token_str = ' '.join(str(x) for x in semantic_token)
            data.append([utt_id, semantic_token_str])
        except Exception:
            print(f"{utt_id} occur Exception")
            traceback.print_exc()
            continue

    delimiter = '\t'
    filename = output_dir / f'semantic_token_{args.rank}_{args.nshard}.tsv'

    with open(filename, 'w', encoding='utf-8') as writer:
        for row in data:
            line = delimiter.join(row)
            writer.write(line + '\n')
    print(f"tsv file '{filename}' write down")
    print(f"time of save stage: {round(time.time() - save_start_time,2)}s")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features 11labs datasets.")

    parser.add_argument(
        "--data_dir", default=None, type=str, help="directory to dataset.")

    parser.add_argument(
        "--dump_dir",
        type=str,
        required=True,
        help="directory to dump feature files.")

    parser.add_argument(
        "--hubert_path", type=str, default='./hubert_base_ls960.pt')

    parser.add_argument(
        "--quantizer_path",
        type=str,
        default='./hubert_base_ls960_L9_km500.bin')

    parser.add_argument(
        "--num-cpu", type=int, default=1, help="number of process.")

    parser.add_argument(
        "--layer",
        type=int,
        default=10,
        help="use which layer of feature of hubert, should be same with it in exp/dump_hubert_feature.py"
    )
    parser.add_argument(
        '--sr', type=int, default=16000, help='sample rate of model')

    # For LibriLight dataset
    parser.add_argument("--nshard", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()

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

    # 每个 speaker 提供 10 条 test 10 条 dev (但如果总量 <100 就不分了)
    # 11 labs 共 112 个 speaker
    # LibriTTS 共 2,456 个 speaker, 每个 speaker 各提供一条 dev 和 test
    train_wav_files = []
    dev_wav_files = []
    test_wav_files = []
    sub_num_dev = 10

    for speaker in speaker_list:
        st = time.time()
        wav_files = sorted(list((data_dir / speaker).rglob("mp3s/*.mp3")))
        print(f"time of rglob and sort: {round(time.time()-st,2)}s")
        # filter out ._*.flac
        wav_files = [
            file for file in wav_files if not file.name.startswith('._')
        ]
        if len(wav_files) > 100:
            train_wav_files += wav_files[:-sub_num_dev * 2]
            dev_wav_files += wav_files[-sub_num_dev * 2:-sub_num_dev]
            test_wav_files += wav_files[-sub_num_dev:]
        else:
            train_wav_files += wav_files

    print(
        f"num of wav files in rank {args.rank} / {args.nshard}: {len(train_wav_files)+len(dev_wav_files) + len(test_wav_files)}"
    )

    print("args.layer:", args.layer)

    train_dump_dir = dump_dir / "train"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(train_dump_dir, 0o777)
    dev_dump_dir = dump_dir / "dev"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(dev_dump_dir, 0o777)
    test_dump_dir = dump_dir / "test"
    test_dump_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(test_dump_dir, 0o777)

    semantic_tokenizer = SemanticTokenizer(
        hubert_path=args.hubert_path,
        quantizer_path=args.quantizer_path,
        duplicate=True,
        output_layer=args.layer)

    # process for the 3 sections
    if train_wav_files:
        process_sentences(
            args=args,
            fps=train_wav_files,
            output_dir=train_dump_dir,
            semantic_tokenizer=semantic_tokenizer,
            nprocs=args.num_cpu)
    if dev_wav_files:
        process_sentences(
            args=args,
            fps=dev_wav_files,
            output_dir=dev_dump_dir,
            semantic_tokenizer=semantic_tokenizer,
            nprocs=args.num_cpu)

    if test_wav_files:
        process_sentences(
            args=args,
            fps=test_wav_files,
            output_dir=test_dump_dir,
            semantic_tokenizer=semantic_tokenizer,
            nprocs=args.num_cpu)


if __name__ == "__main__":
    main()
