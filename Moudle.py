# ----------------------------------------
# Written by Jing Li
# ----------------------------------------
import torch.nn as nn
import torch
import NewMoudle
import torchvision.models as models
from BiLSTM import BiLSTMLayer
import numpy as np
import random

def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False

class moduleNetPart1(nn.Module):
    def __init__(self, hiddenSize, modeType='MSTNet', freeze=True):
        super().__init__()
        self.modeType = modeType

        if 'MSTNet' == modeType:
            self.conv2d = getattr(models, "resnet34")(pretrained=True)
            self.conv2d.fc = NewMoudle.Identity()

            hidden_size = hiddenSize

            self.relu = nn.ReLU(inplace=True)

            self.linear1 = nn.Linear(512, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)

            self.batchNorm1d1 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d2 = nn.BatchNorm1d(hidden_size)
        elif 'CNN+BiLSTM' == modeType:
            self.conv2d = getattr(models, "resnet18")(pretrained=True)
            self.conv2d.fc = NewMoudle.Identity()
        elif 'VAC' == modeType:
            self.conv2d = getattr(models, "resnet18")(pretrained=True)
            self.conv2d.fc = NewMoudle.Identity()

        if freeze:
            freeze_params(self)

    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

    def forward(self, seqData, dataLen):
        if 'MSTNet' == self.modeType:
            len_x = dataLen
            batch, temp, channel, height, width = seqData.shape
            inputs = seqData.reshape(batch * temp, channel, height, width)

            x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])

            n = len(x)
            indices = np.arange(n)
            np.random.shuffle(indices)
            trainIndex = indices[: int(n * 0.5)]
            trainIndex = sorted(trainIndex)
            testIndex = indices[int(n * 0.5):]
            testIndex = sorted(testIndex)

            trainData = x[trainIndex, :, :, :]
            testData = x[testIndex, :, :, :]

            trainData = self.conv2d(trainData)

            with torch.no_grad():
                testData = self.conv2d(testData)

            shape1 = trainData.shape
            shape2 = testData.shape
            x1 = torch.zeros((shape1[0] + shape2[0] , shape1[1])).cuda()

            for i in range(len(trainIndex)):
                x1[trainIndex[i], :] = trainData[i, :]

            for i in range(len(testIndex)):
                x1[testIndex[i], :] = testData[i, :]

            framewise = torch.cat([self.pad(x1[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                                   for idx, lgt in enumerate(len_x)])

            framewise = framewise.reshape(batch, temp, -1)

            framewise = self.linear1(framewise).transpose(1, 2)
            framewise = self.batchNorm1d1(framewise)
            framewise = self.relu(framewise).transpose(1, 2)

            framewise = self.linear2(framewise).transpose(1, 2)
            framewise = self.batchNorm1d2(framewise)
            framewise = self.relu(framewise)
        elif 'CNN+BiLSTM' == self.modeType:
            len_x = dataLen
            batch, temp, channel, height, width = seqData.shape
            inputs = seqData.reshape(batch * temp, channel, height, width)

            x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])

            x1 = self.conv2d(x)

            framewise = torch.cat([self.pad(x1[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                                   for idx, lgt in enumerate(len_x)])

            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        elif 'VAC' == self.modeType:
            len_x = dataLen
            batch, temp, channel, height, width = seqData.shape
            inputs = seqData.reshape(batch * temp, channel, height, width)

            x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])

            x1 = self.conv2d(x)

            framewise = torch.cat([self.pad(x1[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                                   for idx, lgt in enumerate(len_x)])

            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)

        return framewise

class moduleNet_D(nn.Module):
    def __init__(self, hiddenSize, wordSetNum, dataSetName='RWTH', modeType='MSTNet', freeze=True):
        super().__init__()
        self.modeType = modeType
        self.outDim = wordSetNum
        self.logSoftMax = nn.LogSoftmax(dim=-1)

        if 'MSTNet' == modeType:
            self.dataSetName = dataSetName

            hidden_size = hiddenSize
            inputSize = hiddenSize

            self.conv1D1_1 = nn.Conv1d(in_channels=inputSize, out_channels=hidden_size, kernel_size=3, stride=1,
                                       padding=1)
            self.conv1D1_2 = nn.Conv1d(in_channels=inputSize, out_channels=hidden_size, kernel_size=5, stride=1,
                                       padding=2)
            self.conv1D1_3 = nn.Conv1d(in_channels=inputSize, out_channels=hidden_size, kernel_size=7, stride=1,
                                       padding=3)
            self.conv1D1_4 = nn.Conv1d(in_channels=inputSize, out_channels=hidden_size, kernel_size=9, stride=1,
                                       padding=4)

            self.conv2D1 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(4, 2), stride=2,
                                     padding=0)

            self.conv1D2_1 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1,
                                       padding=1)
            self.conv1D2_2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, stride=1,
                                       padding=2)
            self.conv1D2_3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=7, stride=1,
                                       padding=3)
            self.conv1D2_4 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=9, stride=1,
                                       padding=4)

            self.conv2D2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(4, 2), stride=2,
                                     padding=0)

            self.batchNorm1d1_1 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d1_2 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d1_3 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d1_4 = nn.BatchNorm1d(hidden_size)

            self.batchNorm1d2_1 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d2_2 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d2_3 = nn.BatchNorm1d(hidden_size)
            self.batchNorm1d2_4 = nn.BatchNorm1d(hidden_size)

            self.batchNorm2d1 = nn.BatchNorm2d(hidden_size)
            self.batchNorm2d2 = nn.BatchNorm2d(hidden_size)

            self.relu = nn.ReLU(inplace=True)

            heads = 8
            semantic_layers = 2
            dropout = 0
            rpe_k = 8
            self.temporal_model = NewMoudle.TransformerEncoder(hidden_size, heads, semantic_layers, dropout, rpe_k)

            self.classifier1 = nn.Linear(hidden_size, self.outDim)
            self.classifier2 = nn.Linear(hidden_size, self.outDim)
            if self.dataSetName == 'RWTH':
                self.classifier3 = nn.Linear(hidden_size, self.outDim)
                self.classifier4 = nn.Linear(inputSize, self.outDim)
        elif 'CNN+BiLSTM' == modeType:
            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hiddenSize, hidden_size=hiddenSize,
                                              num_layers=2, bidirectional=True)

            self.classifier = nn.Linear(hiddenSize, self.outDim)
        elif 'VAC' == modeType:
            hidden_size = hiddenSize * 2
            self.conv1d = NewMoudle.TemporalConv(input_size=512,
                                                 hidden_size=hidden_size,
                                                 conv_type=2)

            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)

            self.classifier = nn.Linear(hidden_size, self.outDim)
            self.classifier1 = nn.Linear(hidden_size, self.outDim)

        if freeze:
            freeze_params(self)

    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

    def forward(self, seqData, dataLen):
        framewise = seqData
        len_x = dataLen

        if 'MSTNet' == self.modeType:
            inputData = self.conv1D1_1(framewise)
            inputData = self.batchNorm1d1_1(inputData)
            inputData = self.relu(inputData)

            glossCandidate = inputData.unsqueeze(2)

            inputData = self.conv1D1_2(framewise)
            inputData = self.batchNorm1d1_2(inputData)
            inputData = self.relu(inputData)

            tmpData = inputData.unsqueeze(2)
            glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

            inputData = self.conv1D1_3(framewise)
            inputData = self.batchNorm1d1_3(inputData)
            inputData = self.relu(inputData)

            tmpData = inputData.unsqueeze(2)
            glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

            inputData = self.conv1D1_4(framewise)
            inputData = self.batchNorm1d1_4(inputData)
            inputData = self.relu(inputData)

            tmpData = inputData.unsqueeze(2)
            glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

            inputData = self.conv2D1(glossCandidate)
            inputData = self.batchNorm2d1(inputData)
            inputData1 = self.relu(inputData).squeeze(2)

            # 2
            inputData = self.conv1D2_1(inputData1)
            inputData = self.batchNorm1d2_1(inputData)
            inputData = self.relu(inputData)

            glossCandidate = inputData.unsqueeze(2)

            inputData = self.conv1D2_2(inputData1)
            inputData = self.batchNorm1d2_2(inputData)
            inputData = self.relu(inputData)

            tmpData = inputData.unsqueeze(2)
            glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

            inputData = self.conv1D2_3(inputData1)
            inputData = self.batchNorm1d2_3(inputData)
            inputData = self.relu(inputData)

            tmpData = inputData.unsqueeze(2)
            glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

            inputData = self.conv1D2_4(inputData1)
            inputData = self.batchNorm1d2_4(inputData)
            inputData = self.relu(inputData)

            tmpData = inputData.unsqueeze(2)
            glossCandidate = torch.cat([glossCandidate, tmpData], dim=2)

            inputData = self.conv2D2(glossCandidate)
            inputData = self.batchNorm2d2(inputData)
            inputData = self.relu(inputData).squeeze(2)

            if self.dataSetName == 'RWTH':
                lgt = torch.cat(len_x, dim=0) // 4
                x = inputData.permute(0, 2, 1)
            else:
                lgt = (torch.cat(len_x, dim=0) // 4) - 6
                x = inputData.permute(0, 2, 1)
                x = x[:, 3:-3, :]

            outputs = self.temporal_model(x)

            outputs = outputs.permute(1, 0, 2)
            encoderPrediction = self.classifier1(outputs)
            logProbs1 = self.logSoftMax(encoderPrediction)

            outputs = x.permute(1, 0, 2)
            encoderPrediction = self.classifier2(outputs)
            logProbs2 = self.logSoftMax(encoderPrediction)

            if self.dataSetName == 'RWTH':
                outputs = inputData1.permute(2, 0, 1)
                encoderPrediction = self.classifier3(outputs)
                logProbs3 = self.logSoftMax(encoderPrediction)

                outputs = framewise.permute(2, 0, 1)
                encoderPrediction = self.classifier4(outputs)
                logProbs4 = self.logSoftMax(encoderPrediction)
            else:
                logProbs3 = 0
                logProbs4 = 0
        elif 'CNN+BiLSTM' == self.modeType:
            x = framewise.permute(2, 0, 1)
            lgt = torch.cat(len_x, dim=0)

            outputs = self.temporal_model(x, lgt)

            encoderPrediction = self.classifier(outputs['predictions'])
            logProbs1 = self.logSoftMax(encoderPrediction)

            logProbs2 = 0
            logProbs3 = 0
            logProbs4 = 0
        elif 'VAC' == self.modeType:
            conv1d_outputs = self.conv1d(framewise, len_x)
            # x: T, B, C
            x = conv1d_outputs['visual_feat']
            lgt = conv1d_outputs['feat_len']
            x = x.permute(2, 0, 1)
            lgt = torch.cat(lgt, dim=0)

            outputs = self.temporal_model(x, lgt)

            encoderPrediction = self.classifier(outputs['predictions'])
            logProbs1 = self.logSoftMax(encoderPrediction)

            encoderPrediction = self.classifier1(x)
            logProbs2 = self.logSoftMax(encoderPrediction)

            logProbs3 = 0
            logProbs4 = 0
        return logProbs1, logProbs2, logProbs3, logProbs4, lgt

class moduleNet_G(nn.Module):
    def __init__(self, hiddenSize, scale):
        super().__init__()

        self.scale = scale
        self.temporalUpSample = NewMoudle.TemporalUpSample(hiddenSize, scale)

    def forward(self, seqData, isTrain=True):
        framewiseInput = seqData

        if isTrain:
            framewise = framewiseInput[:, :, ::self.scale]

            shape = framewiseInput.shape

            for i in range(shape[2] // self.scale):
                tmp = framewiseInput[:, :, i * self.scale:(i + 1) * self.scale]
                n = random.randint(0, self.scale - 1)
                framewise[:, :, i] = tmp[:, :, n]
        else:
            framewise = framewiseInput

        framewise = self.temporalUpSample(framewise)

        return framewise