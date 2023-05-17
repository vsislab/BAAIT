# 人工智能技术基础及应用

<div align="center">

[English](README_EN.md) | 简体中文

</div>

---

本仓库为VSISLAB所编写《人工智能技术基础及应用》一书配套代码

第1章不涉及实践，无配套代码

## 第二章 神经网络基础

本章主要介绍神经网络的基础设计，使用numpy从零实现神经网络中常用的模块，最终搭建一个可以实现手写数字识别的网络

## 第三章 深度学习计算框架

本章主要介绍PyTorch深度学习计算框架，利用PyTorch以更简洁的方式实现第二章中的网络

## 第四章 卷积神经网络

本章主要介绍经典卷积神经网络结构，训练及测试所用数据集为FashionMNIST，构建Dataloader时会自动下载至目录`chapter_4/data/FashionMNIST`

<div align="center">
  <p>
    <img src="doc/imgs/Fashion-MNIST-dataset.png" width="400">
    <br/>
    FashionMNIST样本
  </p>
</div>


[FashionMNIST官方仓库](https://github.com/zalandoresearch/fashion-mnist)


## 第五章 序列到序列网络

本章主要介绍经典序列到序列网络，训练及测试所用数据集为aclImdb，需要读者下载至目录`chapter_5/data/aclImdb`

<div align="center">
    <p>
      <img src="doc/imgs/aclImdb.png" style="text-align:center" width="500">
      <br/>
      正负评价样本
    </p>
</div>


[aclImdb下载地址](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

## 第六章 目标检测及其应用

本章主要介绍目标分割实践案例，训练即测试所用数据集为Penn-Fudan，在jupyter notebook中有下载脚本

<div align="center">
    <p>
      <img src="doc/imgs/Penn-Fudan.png" style="text-align:center" width="500">
      <br/>
      行人检测
    </p>
</div>

[Penn-Fudan官网地址](https://www.cis.upenn.edu/~jshi/ped_html/)

## 第七章 语义分割及其应用

本章实现城市街景分割实践案例，训练及测试所用数据集为CamVid，需要读者下载至目录`chapter_7/CamVid`

<div align="center">
  <p>
    <img src="doc/imgs/CamVid_0006R0_f01260.png" style="text-align:center" width="500">
    <br/>
    原图像
  </p>
</div>


<div align="center">
  <p>
    <img src="doc/imgs/CamVid_0006R0_f01260_P.png" style="text-align:center" width="500">
    <br/>
    语义分割输出
  </p>
</div>

[CamVid官网地址](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)

[CamVid亚马逊云下载地址](https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz)