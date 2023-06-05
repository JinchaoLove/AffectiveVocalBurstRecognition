# [A Hierarchical Regression Chain Framework for Affective Vocal Burst Recognition](https://arxiv.org/pdf/2303.08027.pdf)

## Info

Our work on [ACII-AVB 2022 Challenge](https://www.competitions.hume.ai/avb2022), **winner** in the tasks of `A-VB Two` and `A-VB Culture`, second in the task of `A-VB High`.


## Files

- `config/`: configurations for data (`data.yaml`), model (`model.yaml`), training (`train.yaml`) and logger `logger.yaml`.
  - `train.lite` configs training environment such as `ddp`.
  - `logger.wandb`: [W&B logger](https://docs.wandb.ai), initialize your API key at the first time if `>0`; will save model in `<logger.dir>/wandb/latest-runn/files` if `>1`.
  - all of the configs of `module`, `optimizer`, `iterator`, `callbacks` can also be passed/overided through `trainer` using `__` (recursively for `callbacks`).
- `filelists/`: splitted filelists for train, validation and test
- `models/`: nn modules, such as upstream, downstream, losses, etc.
- `trainer/`: support wrappers of trainer with loggers and callbacks
- `utils/`: data process, callbacks and metrics
- `cv.py`: cross-validation
- `data_augment.py`: data augmentation
- `dataset.py`: data preparation
- `lite.py`: training wrapper
- `run_exp.sh`, `nex_exp.sh`: run a set of experiments
- `requirements.txt`: auto generated by `pipreqs .` with no strict version specification
- `test.py`: model evaluation
- `train.py`: main training file with config of data, model, callbacks, etc.


## Process
1. Setup environment (generated by pipreqs, python version is 3.9, recommend our [Docker Image](https://github.com/JinchaoLove/Docker_Images))
```bash
conda create -n pt python==3.9.12 pytorch==1.11.0 torchaudio==0.11.0 cudatoolkit -c pytorch -y -q # may need cuda version for `cudatoolkit` (nvcc --version)
conda activate pt
pip install -q -r requirements.txt &
echo "export PYTHONPATH=${PYTHONPATH}:$(pwd)" >> ~/.<shell>rc  # add the path of the workspace
source ~/.<shell>rc  # update shell environment
```

2. Trim silence in wav files
```bash
python3 utils/preprocess.py --src_dir /path/to/wav --tgt_dir /path/to/output/dir
```

3. Create filelists
```bash
python3 utils/create_splits.py --data_dir=/path/to/data --save_path=./filelists
```

## Training
Run the following cmd to train the model.
```bash
python3 train.py
# pkill -f train.py (if stucked)
```

This will train the model with default setting using the model in `models.ssl_trans.MTL`. If you want to train other models or modify the parameters, please refer the config files under the `config` dir.

## Cross-validation

```bash
python3 cv.py
```
This will run cross-validation with default setting.


## Reference

- [The ACII 2022 Affective Vocal Bursts Workshop & Competition: Understanding a critically understudied modality of emotional expression](https://arxiv.org/abs/2207.03572) [Code](https://github.com/HumeAI/competitions/tree/main/A-VB2022)
- [Exploring the Effectiveness of Self-supervised Learning and Classifier Chains in Emotion Recognition of Nonverbal Vocalizations](https://arxiv.org/abs/2206.10695) [Code](https://github.com/Aria-K-Alethia/ExVo)
- Environment supported by [LightningLite](https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/lite/lite.py)
- Trainer modified from [skorch](https://github.com/skorch-dev/skorch/blob/master/skorch/net.py)
- Config supported by [hydra_core](https://github.com/facebookresearch/hydra)


## Authors

Please give me a 🌟 if this repository helps you 🤗

If you have any questions, please feel free to issue or contact me ([Jinchao](http://jinchaoli.com)).