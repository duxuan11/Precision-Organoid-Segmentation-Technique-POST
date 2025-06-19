<h1 align="center">Precision Organoid Segmentation Technique (POST)</h1>

<div align='center'>
    <a href='https://scholar.google.com' target='_blank'><strong>Xuan Du</strong></a><sup> 1,3</sup>,&thinsp;
    <a href='https://scholar.google.com' target='_blank'><strong>YuChen Li</strong></a><sup>1,3</sup>,&thinsp;
    <a href='https://scholar.google.com' target='_blank'><strong>Jiaping Song</strong></a><sup>4</sup>,&thinsp;
    <a href='https://scholar.google.com' target='_blank'><strong>Zilin Zhang</strong></a><sup>1</sup>,&thinsp;
    <a href='https://scholar.google.com' target='_blank'><strong>Jing Zhang</strong></a><sup>4</sup>,&thinsp;
    <a href='https://scholar.google.com' target='_blank'><strong>Yanhui Li</strong></a><sup>2*</sup>,&thinsp;
    <a href='https://scholar.google.com' target='_blank'><strong>Zaozao Chen</strong></a><sup>1*</sup>,&thinsp;
    <a href='https://scholar.google.com' target='_blank'><strong>Zhongze Gu</strong></a><sup>1*</sup>,&thinsp;
</div>

<div align='center'>
    <sup>1 </sup>Southeast University&ensp;  <sup>2 </sup>Nanjing University&ensp;  <sup>3 </sup>Avatarget Biotechnology Co.&ensp;  <br/> <sup>4 </sup>Anhui Science and Technology University&ensp;
    <br />
</div>
<div align="center" style="display: flex; justify-content: center; flex-wrap: wrap;">
  <a href='https://www.sciopen.com/article/pdf/10.26599/AIR.2024.9150'><img src='https://img.shields.io/badge/Journal-Paper-red'></a>&ensp;
  <a href='https://arxiv.org/pdf/2401.'><img src='https://img.shields.io/badge/arXiv-Paper-green'></a>&ensp;
  <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-blue'></a>&ensp;
</div>

|            *Sample_1*            |             *Sample_2*            |             *Sample_3*            |
| :------------------------------: | :-------------------------------: | :-------------------------------: |
| <img src="https://drive.google.com/uc?id=19ShmEUc_lpIKASy6zUq5F89q3HRsdfvr" /> |  <img src="https://drive.google.com/uc?id=1tJEwPgfiK18r34n_qronNpuQhjd20NhV" /> |  <img src="https://drive.google.com/uc?id=1KtoevItyasK2samFIE0RfrQQ7cHo0QBZ" /> |
|            *Sample_4*            |             *Sample_5*            |             *Sample_6*            |
| <img src="https://drive.google.com/uc?id=1I5UW-xzCLMQD0IhlRwdCHqz8nFMxUa8X" /> |  <img src="https://drive.google.com/uc?id=1BbR6YjzI8xKwCQL-t-5pul5PcLRrJNrs" /> |  <img src="https://drive.google.com/uc?id=1puGcK3elZ3__nfKTa2iVcMajLhLaG1qQ" /> |
<br/>
This repo is the official implementation of "[**Precision Organoid Segmentation Technique (POST): an Accurate Algorithm for Segmentation of any Challenging Bright-field Organoid Images.**]

## Documentation 📑
> This is about the organoid/organ-on-a-chip intelligent segmentation algorithm.

## Usage
#### Environment Setup
```shell
conda create -n POST python==3.10.0
pip install -r requirements.txt
```
### Prediction

```shell
python inference.py
```
### Training
You can refer to the training process of [yolov8](https://github.com/ultralytics/ultralytics) and [yolov9](https://github.com/WongKinYiu/yolov9).Then use these trained models in this code.

### Download
Post models:[Google Drive](https://drive.google.com/drive/folders/1-Dd-zFxHM2GfprqbEv2Tv0_mLNu88SuW?usp=sharing)

## License
The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source.

## Acknowledgement
Our project is developed based on [yolov8](https://github.com/ultralytics/ultralytics),[yolov9](https://github.com/WongKinYiu/yolov9), and [TinySAM](https://github.com/xinghaochen/TinySAM).
