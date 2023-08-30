可以先在自己的目录 dump 数据，然后软链接到 LibriLight 的 dump 目录下和 LibriLight 数据一起训练或者 finetune
dump 好的 11labs 目录和 LibriLight 的 small / medium / large / duplicate 保持一致，目录结构可以如下:
```text
dump_librilight
├── 11labs
│   ├── dev
│   ├── test
│   └── train
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
