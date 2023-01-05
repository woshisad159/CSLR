# ----------------------------------------
# Written by Jing Li
# ----------------------------------------
import Moudle
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from WER import WerScore
import os
import DataProcessMoudle
import DecodeMoudle
import videoAugmentation
import numpy as np
import decode
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from evaluation import evaluteMode

def train(configParams, isTrain=True):
    # Parameter initialization
    # Read in data path
    trainDataPath = configParams["trainDataPath"]
    validDataPath = configParams["validDataPath"]
    testDataPath = configParams["testDataPath"]
    # Read in label path
    trainLabelPath = configParams["trainLabelPath"]
    validLabelPath = configParams["validLabelPath"]
    testLabelPath = configParams["testLabelPath"]
    # Read in model parameters
    bestModuleSavePath = configParams["bestModuleSavePath"]
    currentModuleSavePath = configParams["currentModuleSavePath"]
    # Read in parameters
    device = configParams["device"]
    hiddenSize = int(configParams["hiddenSize"])
    lr = float(configParams["lr"])
    batchSize = int(configParams["batchSize"])
    numWorkers = int(configParams["numWorkers"])
    pinmMemory = bool(int(configParams["pinmMemory"]))
    moduleChoice = configParams["moduleChoice"]
    dataSetName = configParams["dataSetName"]
    scale = int(configParams["scale"])
    sourcefilePath = configParams["sourcefilePath"]
    if isTrain:
        fileName = "output-hypothesis-{}.ctm".format('dev')
    else:
        fileName = "output-hypothesis-{}.ctm".format('test')
    filePath = os.path.join(sourcefilePath, fileName)

    # Preprocessing Language Sequences
    word2idx, wordSetNum, idx2word = DataProcessMoudle.Word2Id(trainLabelPath, validLabelPath, testLabelPath, dataSetName)
    # Image preprocessing
    if dataSetName == "RWTH":
        transform = videoAugmentation.Compose([
            videoAugmentation.RandomCrop(224),
            videoAugmentation.RandomHorizontalFlip(0.5),
            videoAugmentation.ToTensor(),
        ])

        transformTest = videoAugmentation.Compose([
            videoAugmentation.CenterCrop(224),
            videoAugmentation.ToTensor(),
        ])
    elif dataSetName == "CSL":
        transform = videoAugmentation.Compose([
            videoAugmentation.RandomCrop(224),
            videoAugmentation.RandomHorizontalFlip(0.5),
            videoAugmentation.ToTensor(),
        ])

        transformTest = videoAugmentation.Compose([
            videoAugmentation.CenterCrop(224),
            videoAugmentation.ToTensor(),
        ])

    # Import Data
    trainData = DataProcessMoudle.MyDataset(trainDataPath, trainLabelPath, word2idx, dataSetName, scale=scale, isTrain=True, transform=transform)

    validData = DataProcessMoudle.MyDataset(validDataPath, validLabelPath, word2idx, dataSetName, scale=scale, transform=transformTest)

    if dataSetName == "RWTH":
        testData = DataProcessMoudle.MyDataset(testDataPath, testLabelPath, word2idx, dataSetName, scale=scale, transform=transformTest)

    trainLoader = DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True, num_workers=numWorkers,
                             pin_memory=pinmMemory, collate_fn=DataProcessMoudle.collate_fn, drop_last=True)
    validLoader = DataLoader(dataset=validData, batch_size=batchSize, shuffle=False, num_workers=numWorkers,
                             pin_memory=pinmMemory, collate_fn=DataProcessMoudle.collate_fn, drop_last=True)
    if dataSetName == "RWTH":
        testLoader = DataLoader(dataset=testData, batch_size=batchSize, shuffle=False, num_workers=numWorkers,
                                pin_memory=pinmMemory, collate_fn=DataProcessMoudle.collate_fn, drop_last=True)
    # Define model
    ########################## Part1 ########################
    moduleNetPart1 = Moudle.moduleNetPart1(hiddenSize, modeType=moduleChoice).to(device)

    path = "module/" + moduleChoice + '_' + dataSetName + '.pth'
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model_now_dict = moduleNetPart1.state_dict()

    load_pretrained_dict = (checkpoint["moduleNet_state_dict"])
    new_state_dict = {k: v for k, v in load_pretrained_dict.items() if k in model_now_dict.keys()}

    # 1. filter out unnecessary keys
    # 2. overwrite entries in the existing state dict
    model_now_dict.update(new_state_dict)
    moduleNetPart1.load_state_dict(model_now_dict)
    ########################## G ########################
    moduleNet = Moudle.moduleNet_G(hiddenSize, scale).to(device)
    ########################## D ########################
    moduleNet_D = Moudle.moduleNet_D(hiddenSize, wordSetNum + 1, dataSetName, modeType=moduleChoice).to(device)

    model_now_dict = moduleNet_D.state_dict()

    new_state_dict = {k: v for k, v in load_pretrained_dict.items() if k in model_now_dict.keys()}

    # 1. filter out unnecessary keys
    # 2. overwrite entries in the existing state dict
    model_now_dict.update(new_state_dict)
    moduleNet_D.load_state_dict(model_now_dict)
    # Definition of loss function
    PAD_IDX = 0
    ctcLoss = nn.CTCLoss(blank=PAD_IDX, reduction='mean', zero_infinity=True).to(device)
    kld = DataProcessMoudle.SeqKD(T=8).to(device)
    logSoftMax = nn.LogSoftmax(dim=-1)
    # Optimization function
    params = list(moduleNet.parameters())

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0001)
    # Read pretraining model parameters
    bestLoss = 65535
    bestLossEpoch = 0
    bestWerScore = 65535
    bestWerScoreEpoch = 0
    epoch = 0
    if os.path.exists(currentModuleSavePath):
        checkpoint = torch.load(currentModuleSavePath, map_location=torch.device('cpu'))
        moduleNet.load_state_dict(checkpoint['moduleNet_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        bestLoss = checkpoint['bestLoss']
        bestLossEpoch = checkpoint['bestLossEpoch']
        bestWerScore = checkpoint['bestWerScore']
        bestWerScoreEpoch = checkpoint['bestWerScoreEpoch']
        epoch = checkpoint['epoch']
        lastEpoch = epoch
        print(f"Pretraining model loaded epoch: {epoch}, bestLoss: {bestLoss:.5f}, bestEpoch: {bestLossEpoch}, werScore: {bestWerScore:.5f}, bestEpoch: {bestWerScoreEpoch}")
    else:
        lastEpoch = -1
        print(f"Pretraining model not loaded epoch: {epoch}, bestLoss: {bestLoss}, bestEpoch: {bestLossEpoch}, werScore: {bestWerScore:.5f}, bestEpoch: {bestWerScoreEpoch}")

    # Set learning rate attenuation rules
    if dataSetName == "CSL":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=[5,10],
                                                         gamma=0.1, last_epoch=lastEpoch)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=[10, 20],
                                                         gamma=0.1, last_epoch=lastEpoch)
    # Decoding parameters
    beam_width = 10
    prune = 0.01
    decodeMode = DecodeMoudle.Model(wordSetNum, max_num_states = 1)

    decoder = decode.Decode(word2idx, wordSetNum + 1, 'beam')

    if dataSetName == "CSL":
        epochNum = 15
    else:
        epochNum = 30

    if isTrain:
        print("Start training the model")
        # 训练模型
        for _ in range(epochNum):
            moduleNetPart1.eval()
            moduleNet_D.eval()

            moduleNet.train()

            scaler = GradScaler()
            loss_value = []
            for Dict in tqdm(trainLoader):
                data = Dict["video"].to(device)
                label = Dict["label"]
                dataLen = Dict["videoLength"]

                targetOutData = [torch.tensor(decodeMode.decoder.expand(yi)).to(device) for yi in label]
                targetLengths = torch.tensor(list(map(len, targetOutData)))
                targetOutData = torch.cat(targetOutData, dim=0).to(device)

                with autocast():
                    framewiseRef = moduleNetPart1(data, dataLen)

                    framewiseFake = moduleNet(framewiseRef, True)

                    logProbs1, logProbs2, logProbs3, logProbs4, lgt = moduleNet_D(framewiseFake, dataLen)

                    if moduleChoice == 'CNN+BiLSTM':
                        loss = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                    elif moduleChoice == 'VAC':
                        loss3 = 25 * kld(logProbs2, logProbs1, use_blank=False)

                        logProbs1 = logSoftMax(logProbs1)
                        logProbs2 = logSoftMax(logProbs2)

                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths).mean()
                        loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths).mean()

                        loss = loss1 + loss2 + loss3
                    elif moduleChoice == 'MSTNet':
                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                        loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths)

                        if dataSetName == "RWTH":
                            loss3 = ctcLoss(logProbs3, targetOutData, lgt * 2, targetLengths)
                            loss4 = ctcLoss(logProbs4, targetOutData, lgt * 4, targetLengths)
                            loss = loss1 + loss2 + loss3 + loss4
                        else:
                            loss = loss1 + loss2

                    optimizer.zero_grad()

                    scaler.scale(loss).backward()

                    scaler.step(optimizer)

                    scaler.update()

                loss_value.append(loss.item())

                torch.cuda.empty_cache()

            print("epoch: %d, trainLoss: %f, lr : %f" % (
            epoch, np.mean(loss_value), optimizer.param_groups[0]['lr']))

            epoch = epoch + 1

            scheduler.step()
            optimizer.zero_grad()

            moduleNet.eval()
            print("Start validating model")
            # 验证模型
            werScoreSum = 0
            loss_value = []
            total_info = []
            total_sent = []

            for Dict in tqdm(validLoader):
                data = Dict["video"].to(device)
                label = Dict["label"]
                dataLen = Dict["videoLength"]
                info = Dict["info"]

                targetOutData = [torch.tensor(decodeMode.decoder.expand(yi)).to(device) for yi in label]
                targetLengths = torch.tensor(list(map(len, targetOutData)))
                targetData = targetOutData
                targetOutData = torch.cat(targetOutData, dim=0).to(device)
                batchSize = len(targetLengths)

                with torch.no_grad():
                    framewiseRef = moduleNetPart1(data, dataLen)

                    framewiseFake = moduleNet(framewiseRef, False)

                    dataLen1 = [i * scale for i in dataLen]

                    logProbs1, logProbs2, logProbs3, logProbs4, lgt = moduleNet_D(framewiseFake, dataLen1)

                    if moduleChoice == 'CNN+BiLSTM':
                        loss = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                    elif moduleChoice == 'VAC':
                        loss3 = 25 * kld(logProbs2, logProbs1, use_blank=False)

                        logProbs1 = logSoftMax(logProbs1)
                        logProbs2 = logSoftMax(logProbs2)

                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths).mean()
                        loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths).mean()

                        loss = loss1 + loss2 + loss3
                    elif moduleChoice == 'MSTNet':
                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                        loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths)

                        if dataSetName == "RWTH":
                            loss3 = ctcLoss(logProbs3, targetOutData, lgt * 2, targetLengths)
                            loss4 = ctcLoss(logProbs4, targetOutData, lgt * 4, targetLengths)
                            loss = loss1 + loss2 + loss3 + loss4
                        else:
                            loss = loss1 + loss2

                loss_value.append(loss.item())

                if dataSetName == "RWTH":
                    pred = decoder.decode(logProbs1, lgt, batch_first=False, probs=False)

                    total_info += info
                    total_sent += pred
                else:
                    prob = []
                    P = logProbs1.permute(1, 0, 2)
                    prob += [lpi.exp().cpu().numpy() for lpi in P]
                    gloss_id = decodeMode.decode(prob, beam_width, prune)

                    werScore = WerScore(gloss_id, targetData, idx2word, batchSize)
                    werScoreSum = werScoreSum + werScore

                torch.cuda.empty_cache()

            currentLoss = np.mean(loss_value)

            werScore = werScoreSum / len(validLoader)

            if werScore < bestWerScore:
                bestWerScore = werScore
                bestWerScoreEpoch = epoch - 1

                moduleDict = {}
                moduleDict['moduleNet_state_dict'] = moduleNet.state_dict()
                moduleDict['optimizer_state_dict'] = optimizer.state_dict()
                moduleDict['bestLoss'] = bestLoss
                moduleDict['bestLossEpoch'] = bestLossEpoch
                moduleDict['bestWerScore'] = bestWerScore
                moduleDict['bestWerScoreEpoch'] = bestWerScoreEpoch
                moduleDict['epoch'] = epoch
                torch.save(moduleDict, bestModuleSavePath)

            bestLoss = currentLoss
            bestLossEpoch = epoch - 1

            moduleDict = {}
            moduleDict['moduleNet_state_dict'] = moduleNet.state_dict()
            moduleDict['optimizer_state_dict'] = optimizer.state_dict()
            moduleDict['bestLoss'] = bestLoss
            moduleDict['bestLossEpoch'] = bestLossEpoch
            moduleDict['bestWerScore'] = bestWerScore
            moduleDict['bestWerScoreEpoch'] = bestWerScoreEpoch
            moduleDict['epoch'] = epoch
            torch.save(moduleDict, currentModuleSavePath)

            moduleSavePath1 = 'module/bestMoudleNet_' + str(epoch) + '.pth'
            torch.save(moduleDict, moduleSavePath1)

            if dataSetName == "RWTH":
                ##########################################################################
                DataProcessMoudle.write2file(filePath, total_info, total_sent)
                evaluteMode('evalute_dev')
                ##########################################################################
                DataProcessMoudle.write2file('./wer/' + "output-hypothesis-{}{:0>4d}.ctm".format('dev', epoch),
                                             total_info, total_sent)

            print(f"validLoss: {currentLoss:.5f}, werScore: {werScore:.5f}")
            print(f"bestLoss: {bestLoss:.5f}, beatEpoch: {bestLossEpoch}, bestWerScore: {bestWerScore:.5f}, bestWerScoreEpoch: {bestWerScoreEpoch}")
    else:
        for i in range(epochNum):
            currentModuleSavePath = "module/bestMoudleNet_" + str(i + 1) + ".pth"
            checkpoint = torch.load(currentModuleSavePath, map_location=torch.device('cpu'))
            moduleNet.load_state_dict(checkpoint['moduleNet_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            bestLoss = checkpoint['bestLoss']
            bestLossEpoch = checkpoint['bestLossEpoch']
            bestWerScore = checkpoint['bestWerScore']
            bestWerScoreEpoch = checkpoint['bestWerScoreEpoch']

            moduleNetPart1.eval()
            moduleNet_D.eval()
            moduleNet.eval()

            print("Start testing the model")
            # 验证模型
            werScoreSum = 0
            loss_value = []
            total_info = []
            total_sent = []

            for Dict in tqdm(testLoader):
                data = Dict["video"].to(device)
                label = Dict["label"]
                dataLen = Dict["videoLength"]
                info = Dict["info"]

                targetOutData = [torch.tensor(decodeMode.decoder.expand(yi)).to(device) for yi in label]
                targetLengths = torch.tensor(list(map(len, targetOutData)))
                targetData = targetOutData
                targetOutData = torch.cat(targetOutData, dim=0).to(device)
                dataLen1 = [i * scale for i in dataLen]

                with torch.no_grad():
                    framewiseRef = moduleNetPart1(data, dataLen)

                    framewiseFake = moduleNet(framewiseRef, False)

                    logProbs1, logProbs2, logProbs3, logProbs4, lgt = moduleNet_D(framewiseFake, dataLen1)

                    if moduleChoice == 'CNN+BiLSTM':
                        loss = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                    elif moduleChoice == 'VAC':
                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                        loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths)
                        loss = loss1 + loss2
                    elif moduleChoice == 'MSTNet':
                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                        loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths)

                        if dataSetName == "RWTH":
                            loss3 = ctcLoss(logProbs3, targetOutData, lgt * 2, targetLengths)
                            loss4 = ctcLoss(logProbs4, targetOutData, lgt * 4, targetLengths)
                            loss = loss1 + loss2 + loss3 + loss4
                        else:
                            loss = loss1 + loss2

                loss_value.append(loss.item())

                if dataSetName == "RWTH":
                    pred = decoder.decode(logProbs1, lgt, batch_first=False, probs=False)

                    total_info += info
                    total_sent += pred
                elif dataSetName == "CSL":
                    prob = []
                    P = logProbs1.permute(1, 0, 2)
                    prob += [lpi.exp().cpu().numpy() for lpi in P]
                    targetOutDataCTC = decodeMode.decode(prob, beam_width, prune)

                    werScore = WerScore(targetOutDataCTC, targetData, idx2word, batchSize)
                    werScoreSum = werScoreSum + werScore

                torch.cuda.empty_cache()

            currentLoss = np.mean(loss_value)

            werScore = werScoreSum / len(validLoader)

            if dataSetName == "RWTH":
                ##########################################################################
                DataProcessMoudle.write2file(filePath, total_info, total_sent)
                evaluteMode('evalute_test')
                ##########################################################################
                DataProcessMoudle.write2file('./wer/' + "output-hypothesis-{}{:0>4d}.ctm".format('test', i + 1), total_info,
                                             total_sent)

            print(f"validLoss: {currentLoss:.5f}, werScore: {werScore:.5f}")
            print(f"bestLoss: {bestLoss:.5f}, beatEpoch: {bestLossEpoch}, bestWerScore: {bestWerScore:.5f}, bestWerScoreEpoch: {bestWerScoreEpoch}")


