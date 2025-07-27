### Source code for _Polyommatus spp._ image classification

#### Prerequisites
* [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
* [make](https://www.gnu.org/software/make/manual/make.html)

##### Environment setup
```bash
  make env
  conda activate bio
  make setup
```

##### Metadata tables

Place images into `./data` directory and execute the following command:
```bash
  make metadata
```
Metadata tables will be saved into `./data` directory


##### Fine-tune classifier

> Available: resnet50, mobilenet_v3_large, efficientnet_v2_l
```bash
# CLI info: python ./src/train_cls.py --help
  python ./src/train_cls.py \
      --model=<> --epochs=<> --view=<> --batch_size=<> -a \
      --train_size=0.8 --val_size=0.1 --test_size=0.1 --sampler cbalanced \
      --metadata_path=./data/meta_all_groups_v3.csv \
      --dirname=./weights_cls/<> \
      --neptune --tags <>
```

#### Evaluate classifier
```bash
# CLI info: python ./src/eval_cls.py --help
python ./src/eval_cls.py --model <> --view <> \
    --train_size=0.8 --val_size=0.1 --test_size=0.1 \
    --metadata_path=./data/meta_all_groups_v3.csv \
    --dirname ./weights_cls/<> \
    --output_dir=./eval_output
```
