import csv
import os
import torch
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
import cv2
import random
import torch.nn.functional as F
import torch.nn as nn

PAD = ' '

def Word2Id(trainLabelPath, validLabelPath, testLabelPath, dataSetName):
    if dataSetName == "RWTH":
        wordList = []
        with open(trainLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    rowStrList = row[0].split("|")
                    words = rowStrList[3].split()
                    wordList += words

        with open(validLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    rowStrList = row[0].split("|")
                    words = rowStrList[3].split()
                    wordList += words

        with open(testLabelPath, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
                if n != 0:
                    rowStrList = row[0].split("|")
                    words = rowStrList[3].split()
                    wordList += words

    elif dataSetName == "CSL":
        with open(trainLabelPath, "r", encoding="utf-8") as f:
            sourceStr = f.read()

        txtStr = sourceStr.split()[1::2]

        for i, s in enumerate(txtStr):
            txtStr[i] = s.strip("\ufeff")

        wordList = ''.join((x for x in txtStr))

    idx2word = [PAD]
    set2list = sorted(list(set(wordList)))
    idx2word.extend(set2list)

    word2idx = {w: i for i, w in enumerate(idx2word)}

    return word2idx, len(idx2word) - 1, idx2word


class MyDataset(Dataset):
    def __init__(self, ImagePath, LabelPath, word2idx, dataSetName, scale=2, isTrain=False, transform=None):
        """
        path : 数据路径，包含了图像的路径
        transform：数据处理，对图像进行随机剪裁，以及转换成tensor
        """
        self.ImagePath = ImagePath
        self.transform = transform
        self.dataSetName = dataSetName
        self.p_drop = 0.5
        self.random_drop = True
        self.isTrain = isTrain
        self.scale = scale

        if dataSetName == "RWTH":
            lableDict = {}
            with open(LabelPath, 'r', encoding="utf-8") as f:
                reader = csv.reader(f)
                for n, row in enumerate(reader):
                    if n != 0 and n != 2391:
                        rowStrList = row[0].split("|")
                        lableDict[rowStrList[0]] = rowStrList[-1]

            lable = {}
            for line in lableDict:
                sentences = lableDict[line].split()

                txtInt = []
                for i in sentences:
                    txtInt.append(word2idx[i])

                lable[line] = txtInt

            fileName = sorted(os.listdir(ImagePath))

            imgs = []
            for name in fileName:
                try:
                    imageSeqPath = os.path.join(ImagePath, name)
                    imgs.append((imageSeqPath, lable[name]))  # 路径和标签添加到列表中
                except:
                    print(name)

        elif dataSetName == "CSL":
            lableDict = {}
            with open(LabelPath, 'r', encoding="utf-8") as f:
                reader = csv.reader(f)
                for n, row in enumerate(reader):
                    rowStrList = row[0].split()
                    lableDict[rowStrList[0]] = rowStrList[-1]

            lable = {}
            for line in lableDict:
                sentences = lableDict[line]
                sentences = sentences.strip("\ufeff")

                txtInt = []
                for i in sentences:
                    try:
                        txtInt.append(word2idx[i])
                    except:
                        print(sentences)

                lable[line] = txtInt

            fileName = sorted(os.listdir(ImagePath))

            imgs = []
            for name in fileName:
                imageSeqPath = os.path.join(ImagePath, name)

                ImageSeq = sorted(os.listdir(imageSeqPath))

                for i in ImageSeq:
                    frames = os.path.join(imageSeqPath, i)
                    imgs.append((frames, lable[name]))  # 路径和标签添加到列表中

        self.imgs = imgs

    def sample_indices(self, n):
        if self.dataSetName == "RWTH":
            n = (n // 4) * 4
            indices = np.linspace(0, n - 1, num=int(n // 1), dtype=int)
        elif self.dataSetName == "CSL":
            if self.isTrain:
                min_len = 32
                max_len = 230
                L = 1.0 - 0.2
                U = 1.0 + 0.2
                vid_len = n
                new_len = int(vid_len * (L + (U - L) * np.random.random()))
                if new_len < min_len:
                    new_len = min_len
                if new_len > max_len:
                    new_len = max_len
                if (new_len - 4) % 4 != 0:
                    new_len += 4 - (new_len - 4) % 4

                if new_len <= vid_len:
                    indices = sorted(random.sample(range(vid_len), new_len))
                else:
                    indices = sorted(random.choices(range(vid_len), k=new_len))
            else:
                n = (n // 4) * 4
                indices = np.linspace(0, n - 1, num=int(n // 1), dtype=int)

        return indices

    def __getitem__(self, index):
        fn, label = self.imgs[index]  # 通过index索引返回一个图像路径fn 与 标签label
        if self.dataSetName == "RWTH":
            info = fn.split("/")[-1]
            imageSeqPath = os.path.join(fn, "1")

            ImageSeq = sorted(os.listdir(imageSeqPath))

            indices = self.sample_indices(len(ImageSeq))

            frames = [os.path.join(imageSeqPath, i) for i in ImageSeq]
            frames = [frames[i] for i in indices]

            imgSeq = [cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (256, 256)) for img_path in frames]
            imgSeq = self.transform(imgSeq)
            imgSeq = imgSeq.float() / 127.5 - 1

            framewiseInput = imgSeq

            if self.isTrain:
                imgSeq = framewiseInput
            else:
                imgSeq = framewiseInput[::self.scale, :, :, :]

        elif self.dataSetName == "CSL":
            # 图片序列
            info = fn.split("\\")[-1]

            ImagePath = fn

            ImageSeq = sorted(os.listdir(fn))
            # 读取多序列图片
            for i, image in enumerate(ImageSeq):
                imageSeqPath = os.path.join(ImagePath, image)
                ImgSeq = cv2.cvtColor(cv2.imread(imageSeqPath), cv2.COLOR_BGR2RGB)
                seqShape = ImgSeq.shape
                frames1 = ImgSeq.reshape((seqShape[0] // seqShape[1], seqShape[1], seqShape[1], seqShape[2]))
                if i != 0:
                    frames = np.concatenate((frames, frames1), axis=0)
                else:
                    frames = frames1

            imageSize = 256
            imgSeq = [cv2.resize(img, (imageSize, imageSize)) for img in frames]

            indices = self.sample_indices(len(imgSeq)//2)

            imgSeq = [imgSeq[i*2] for i in indices]

            imgSeq = self.transform(imgSeq)
            imgSeq = imgSeq.float() / 127.5 - 1

            framewiseInput = imgSeq
            if self.isTrain:
                imgSeq = framewiseInput
            else:
                imgSeq = framewiseInput[::self.scale, :, :, :]

        sample = {"video":imgSeq,"label":label, "info": info}

        return sample  # 这就返回一个样本

    def __len__(self):
        return len(self.imgs)  # 返回长度，index就会自动的指导读取多少

class defaultdict_with_warning(defaultdict):
    warned = set()
    warning_enabled = False

    def __getitem__(self, key):
        if key == "text" and key not in self.warned and self.warning_enabled:
            print(
                'Warning: using batch["text"] to obtain label is deprecated, '
                'please use batch["label"] instead.'
            )
            self.warned.add(key)

        return super().__getitem__(key)

def collate_fn(batch):
    collated = defaultdict_with_warning(list)

    batch = [item for item in sorted(batch, key=lambda x: len(x["video"]), reverse=True)]

    max_len = len(batch[0]["video"])

    padded_video = []
    for sample in batch:
        vid = sample["video"]
        collated["videoLength"].append(torch.LongTensor([len(vid)]))
        padded_video.append(torch.cat(
            (
                vid,
                vid[-1][None].expand(max_len - len(vid), -1, -1, -1),
            )
            , dim=0))
        collated["label"].append(torch.tensor(sample["label"]).long())
        collated["info"].append(sample["info"])
    padded_video = torch.stack(padded_video)
    collated["video"] = padded_video
    collated.warning_enabled = True

    return dict(collated)

def RemoveBlank(labels, maxSentenceLen, blank=0):
    new_labels = []
    # 合并相同的标签
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # 删除blank
    new_labels = [l for l in new_labels if l != blank]

    if len(new_labels) < maxSentenceLen:
        for _ in range(maxSentenceLen - len(new_labels)):
            new_labels.append(0)
        new_labelsTmp = new_labels
    else:
        new_labelsTmp = new_labels[:maxSentenceLen]

    outPut = torch.Tensor(new_labelsTmp)

    return outPut

def CTCGreedyDecode(y, maxSentenceLen, blank=0):
    # 按列取最大值，即每个时刻t上最大值对应的下标
    raw_rs = y.argmax(dim=-1)
    # 移除blank,值为0的位置表示这个位置是blank
    rs = RemoveBlank(raw_rs, maxSentenceLen, blank)
    return rs

def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))

class SeqKD(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, T=1):
        super(SeqKD, self).__init__()
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        self.T = T

    def forward(self, prediction_logits, ref_logits, use_blank=True):
        start_idx = 0 if use_blank else 1
        prediction_logits = F.log_softmax(prediction_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        ref_probs = F.softmax(ref_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        loss = self.kdloss(prediction_logits, ref_probs)*self.T*self.T
        return loss