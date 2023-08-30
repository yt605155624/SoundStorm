import argparse
import os
import time
import traceback
from academicodec.models.encodec.net3 import SoundStream
from academicodec.models.encodec.test import remove_encodec_weight_norm
from academicodec.models.hificodec.vqvae import VQVAE
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch
import tqdm
from soundstorm.s2.exps.hubert.feature_utils import get_shard_range


# 损坏的 numpy 会重新生成
def check_numpy_file(file_path):
    try:
        # 尝试加载 numpy 文件
        np.load(file_path)
        # print("文件存在且没有损坏。")
        return True
    except Exception:
        # traceback.print_exc()
        print(f'Cannot load {file_path}, will return False and regenerate it')
        return False
    return False


def process_sentence(args, fp: Path, output_dir: Path, codec_extractor):
    utt_id = fp.stem
    sr = args.sr
    record = None

    acoustic_token_dir = output_dir / "acoustic_token" / args.codec_name
    acoustic_token_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(acoustic_token_dir, 0o777)

    try:
        acoustic_token_path = acoustic_token_dir / (utt_id + ".npy")
        if os.path.exists(acoustic_token_path) and check_numpy_file(
                acoustic_token_path):
            # print(acoustic_token_path, 'exits!')
            pass
        else:
            # wav.shape (T, )
            wav, _ = librosa.load(str(fp), sr=args.sr)
            # wav.shape (1, T)
            wav = torch.tensor(wav).unsqueeze(0)
            wav = wav.cuda()
            if args.codec_name == 'hificodec':
                # (1, T, 4)
                acoustic_token = codec_extractor.encode(wav)
                # trans acoustic_token.shape to (Nq, T)
                acoustic_token = acoustic_token.squeeze(0).transpose(0, 1)
            elif args.codec_name == 'encodec':
                # wav.shape (1, 1, T)
                wav = wav.unsqueeze(1)
                # (24, 1, T)
                acoustic_token = codec_extractor.encode(
                    wav, target_bw=args.target_bw)
                # trans acoustic_token.shape to (Nq, T)
                acoustic_token = acoustic_token.squeeze(1)
            else:
                print("Please input the right codec_name!")

            acoustic_token_np = acoustic_token.detach().cpu().numpy()
            np.save(acoustic_token_path, acoustic_token_np)
            # -rw-r--r-- => rw-rw-rw-
            os.chmod(acoustic_token_path, 0o666)
        record = {"utt_id": utt_id, "acoustic_token_path": acoustic_token_path}
    except Exception:
        print("occur Exception")
        traceback.print_exc()
        return None
    return record


def process_sentences(args,
                      fps: List[Path],
                      output_dir: Path,
                      codec_extractor,
                      nprocs: int=1):
    if nprocs == 1:
        results = []
        for fp in tqdm.tqdm(fps, total=len(fps)):
            record = process_sentence(
                args=args,
                fp=fp,
                output_dir=output_dir,
                codec_extractor=codec_extractor)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp in fps:
                    future = pool.submit(process_sentence, args, fp, output_dir,
                                         codec_extractor)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)
    # torch.save() to a large `.pth` file
    acoustic_token_dict = {}
    # record 是 List of Dict, 一条大 wav 一个 record，一条小 wav 一个 sub_recored
    print(f"start to save {args.rank}_{args.nshard}.pth ...")
    save_start_time = time.time()
    for record in tqdm.tqdm(results, total=len(results), colour='green'):
        try:
            utt_id = record["utt_id"]
            acoustic_token_np = np.load(record["acoustic_token_path"])
            acoustic_token = torch.tensor(acoustic_token_np)
            acoustic_token_dict[utt_id] = acoustic_token
        except Exception:
            print(f"{utt_id} occur Exception")
            traceback.print_exc()
            continue

    filename = output_dir / "acoustic_token" / f'{args.codec_name}_{args.rank}_{args.nshard}.pth'
    torch.save(acoustic_token_dict, filename)
    print(f"pth file '{filename}' write down")

    print(f"time of save stage: {round(time.time() - save_start_time,2)}s")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features for LibriLight.")

    parser.add_argument(
        "--codec_name",
        default="hificodec",
        type=str,
        help="name of codec, should in {hificodec, encodec} now")

    parser.add_argument(
        "--data_dir", default=None, type=str, help="directory to dataset.")

    parser.add_argument(
        "--dump_dir",
        type=str,
        required=True,
        help="directory to dump feature files.")

    parser.add_argument(
        "--model_path", type=str, default='./HiFi-Codec-16k-320d')

    parser.add_argument(
        '--sr', type=int, default=16000, help='sample rate of model')

    # for HiFi-Codec
    parser.add_argument(
        "--config_path", type=str, default='./config_16k_320d.json')

    # for Encodec
    parser.add_argument(
        '--ratios',
        type=int,
        nargs='+',
        # probs(ratios) = hop_size, default for 16k_320d
        default=[8, 5, 4, 2],
        help='ratios of SoundStream / Encodec, shoud be set for different hop_size (32d, 320, 240d, ...)'
    )
    parser.add_argument(
        '--target_bandwidths',
        type=float,
        nargs='+',
        # default for 16k_320d
        default=[1, 1.5, 2, 4, 6, 12],
        help='target_bandwidths of net3.py')
    parser.add_argument(
        '--target_bw',
        type=float,
        # default for 16k_320d
        default=12,
        help='target_bw of net3.py')

    parser.add_argument(
        "--num-cpu", type=int, default=1, help="number of process.")

    parser.add_argument("--nshard", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    dump_dir = Path(args.dump_dir).expanduser()
    # use absolute path
    dump_dir = dump_dir.resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)

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

    train_dump_dir = dump_dir / "train"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(train_dump_dir, 0o777)
    dev_dump_dir = dump_dir / "dev"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(dev_dump_dir, 0o777)
    test_dump_dir = dump_dir / "test"
    test_dump_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(test_dump_dir, 0o777)

    if args.codec_name == 'hificodec':
        model = VQVAE(
            config_path=args.config_path,
            ckpt_path=args.model_path,
            with_encoder=True)
        model.cuda()
        model.generator.remove_weight_norm()
        model.encoder.remove_weight_norm()
        model.eval()

    elif args.codec_name == 'encodec':
        model = SoundStream(
            n_filters=32,
            D=512,
            ratios=args.ratios,
            sample_rate=args.sr,
            target_bandwidths=args.target_bandwidths)
        parameter_dict = torch.load(args.model_path)
        new_state_dict = {}
        # k 为 module.xxx.weight, v 为权重
        for k, v in parameter_dict.items():
            # 截取 `module.` 后面的 xxx.weight
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.cuda()
        remove_encodec_weight_norm(model)
        model.eval()

    else:
        print("Please input the right codec_name!")

    codec_extractor = model

    # process for the 3 sections
    if train_wav_files:
        process_sentences(
            args=args,
            fps=train_wav_files,
            output_dir=train_dump_dir,
            codec_extractor=codec_extractor,
            nprocs=args.num_cpu)
    if dev_wav_files:
        process_sentences(
            args=args,
            fps=dev_wav_files,
            output_dir=dev_dump_dir,
            codec_extractor=codec_extractor,
            nprocs=args.num_cpu)
    if test_wav_files:
        process_sentences(
            args=args,
            fps=test_wav_files,
            output_dir=test_dump_dir,
            codec_extractor=codec_extractor,
            nprocs=args.num_cpu)


if __name__ == "__main__":
    main()
