
run.sh

# 数据预处理的时候用这个
dump_dir=dump_11labs
# 训练的时候用这个，并且把 dump_11labs 软链接进来
dump_dir=dump_librilight

```bash
cd dump_librilight
ln -snf ../dump_11labs 11labs
```

tree -L 2

```text
.
├── 11labs -> ../dump_11labs
│    ├── dev
│    ├── test
│    └── train
├── duplicate
│   ├── dev
│   ├── test
│   └── train
├── large
│   ├── dev
│   ├── test
│   └── train
├── medium
│   ├── dev
│   ├── test
│   └── train
└── small
    ├── dev
    ├── test
    └── train
```
第 3 级子目录中是 `phonemes_*.npy` 和软链接过来的 `semantic_token_*.tsv`
