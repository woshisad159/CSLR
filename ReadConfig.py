# ----------------------------------------
# Written by Jing Li
# ----------------------------------------
import configparser
import os
import torch

def readConfig():
    # Default Config Parameters
    configParams = {
        "trainDataPath":"data/RWTH/train",
        "validDataPath": "data/RWTH/valid",
        "testDataPath": "data/RWTH/test",
        "trainLabelPath": "data/RWTH/train.corpus.csv",
        "validLabelPath": "data/RWTH/dev.corpus.csv",
        "testLabelPath": "data/RWTH/test.corpus.csv",
        "bestModuleSavePath": "module/bestMoudleNet.pth",
        "currentModuleSavePath": "module/currentMoudleNet.pth",
        "sourcefilePath": "evaluation/wer/evalute",
        "device": 1, # 0:CPU  1:GPU
        "hiddenSize":512,
        "lr": 0.1,
        "batchSize": 1,
        "numWorkers": 2,
        "pinmMemory": 1,
        'moduleChoice': 'MSTNet',
        "dataSetName": "RWTH",
        "scale": "4",
    }

    configPath = "params/config.ini"
    if os.path.exists(configPath):
        print("Start reading configuration parameters")
        cf = configparser.ConfigParser()
        cf.read(configPath)  # Read the configuration file. If you write the absolute path of the file, you can not use the OS module

        # Read path parameters
        configParams["trainDataPath"] = cf.get("Path", "trainDataPath")
        configParams["validDataPath"] = cf.get("Path", "validDataPath")
        configParams["testDataPath"] = cf.get("Path", "testDataPath")
        configParams["trainLabelPath"] = cf.get("Path", "trainLabelPath")
        configParams["validLabelPath"] = cf.get("Path", "validLabelPath")
        configParams["testLabelPath"] = cf.get("Path", "testLabelPath")
        configParams["bestModuleSavePath"] = cf.get("Path", "bestModuleSavePath")
        configParams["currentModuleSavePath"] = cf.get("Path", "currentModuleSavePath")
        configParams["sourcefilePath"] = cf.get("Path", "sourcefilePath")
        # Read numerical parameters
        configParams["device"] = cf.get("Params", "device")
        configParams["hiddenSize"] = cf.get("Params", "hiddenSize")
        configParams["lr"] = cf.get("Params", "lr")
        configParams["batchSize"] = cf.get("Params", "batchSize")
        configParams["numWorkers"] = cf.get("Params", "numWorkers")
        configParams["pinmMemory"] = cf.get("Params", "pinmMemory")
        configParams["moduleChoice"] = cf.get("Params", "moduleChoice")
        configParams["dataSetName"] = cf.get("Params", "dataSetName")
        configParams["scale"] = cf.get("Params", "scale")

        print("GPU is %s" % torch.cuda.is_available())
        if 1 == int(configParams["device"]):
            configParams["device"] = torch.device("cuda:0")
        else:
            configParams["device"] = torch.device("cpu")
    else:
        print("The configuration file does not exist %s" % (configPath))
        print("Use default parameters")

    for key in configParams:
        print("%s: %s" %(key, configParams[key]))

    return configParams
