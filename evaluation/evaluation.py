import os
from python_wer_evaluation import wer_calculation
import sys

def evaluate2(mode="dev", evaluate_prefix=None,
             output_file=None):
    '''
    TODO  change file save path
    '''
    os.system(f"bash evaluation/preprocess.sh evaluation/{output_file} evaluation/tmp.ctm evaluation/tmp2.ctm")
    os.system(f"cat evaluation/{evaluate_prefix}-{mode}.stm | sort  -k1,1 > evaluation/tmp.stm")
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    os.system(f"python evaluation/mergectmstm.py evaluation/tmp2.ctm evaluation/tmp.stm")
    os.system(f"cp evaluation/tmp2.ctm evaluation/out.{output_file}")

    ret = wer_calculation(f"evaluation/{evaluate_prefix}-{mode}.stm", f"evaluation/out.{output_file}")

    os.system(f"rm -rf evaluation/out.{output_file}")
    return ret

def evaluate1(mode="dev", evaluate_prefix=None,
             output_file=None):
    '''
    TODO  change file save path
    '''
    os.system(f"bash preprocess.sh {output_file} tmp.ctm tmp2.ctm")
    os.system(f"cat {evaluate_prefix}-{mode}.stm | sort  -k1,1 > tmp.stm")
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    os.system(f"python mergectmstm.py tmp2.ctm tmp.stm")
    os.system(f"cp tmp2.ctm out.{output_file}")

    ret = wer_calculation(f"{evaluate_prefix}-{mode}.stm", f"out.{output_file}")

    os.system(f"rm -rf out.{output_file}")
    return ret

def evaluteMode(mode="dev"):
    if mode == 'dev':
        filePath = "./wer/dev/"
        path = 'wer.txt'

        fileReader = open(path, "w")

        fileList = os.listdir(filePath)
        fileList.sort(key=lambda x: int(x[21:25]))

        werResultList = []
        fileNameList = []
        for i, fileName in enumerate(fileList):
            path = os.path.join(filePath, fileName)
            os.system(f"cp {path} {fileName}")

            ret = evaluate1(
                        mode=mode, output_file=fileName,
                        evaluate_prefix='phoenix2014-groundtruth',
                )

            werResultList.append(ret)
            fileNameList.append(fileName)

            os.system(f"rm -rf {fileName}")

            fileReader.writelines(
                "{} {} {:.2f}\n".format(i,fileName,ret))

        indexValue = werResultList.index(min(werResultList))

        fileReader.writelines(
            "Min WER:index:{}, fileName:{}, WER:{:.2f}\n".format(indexValue, fileNameList[indexValue], werResultList[indexValue]))

        print("Min WER:index:{}, fileName:{}, WER:{:.2f}".format(indexValue, fileNameList[indexValue], werResultList[indexValue]))
    elif mode == 'test':
        filePath = "./wer/test/"
        path = 'wer.txt'

        fileReader = open(path, "w")

        fileList = os.listdir(filePath)
        fileList.sort(key=lambda x: int(x[22:26]))

        werResultList = []
        fileNameList = []
        for i, fileName in enumerate(fileList):
            path = os.path.join(filePath, fileName)
            os.system(f"cp {path} {fileName}")

            ret = evaluate1(
                mode=mode, output_file=fileName,
                evaluate_prefix='phoenix2014-groundtruth',
            )

            werResultList.append(ret)
            fileNameList.append(fileName)

            os.system(f"rm -rf {fileName}")

            fileReader.writelines(
                "{} {} {:.2f}\n".format(i, fileName, ret))

        indexValue = werResultList.index(min(werResultList))

        fileReader.writelines(
            "Min WER:index:{}, fileName:{}, WER:{:.2f}\n".format(indexValue, fileNameList[indexValue],
                                                                 werResultList[indexValue]))

        print("Min WER:index:{}, fileName:{}, WER:{:.2f}".format(indexValue, fileNameList[indexValue],
                                                                 werResultList[indexValue]))
    elif mode == 'evalute_dev':
        path = "evaluation/wer/evalute/output-hypothesis-dev.ctm"
        fileName = "output-hypothesis-dev.ctm"
        os.system(f"cp {path} evaluation/{fileName}")

        mode = 'dev'
        ret = evaluate2(
            mode=mode, output_file=fileName,
            evaluate_prefix='phoenix2014-groundtruth',
        )

        os.system(f"rm -rf evaluation/{fileName}")
    elif mode == 'evalute_test':
        path = "evaluation/wer/evalute/output-hypothesis-test.ctm"
        fileName = "output-hypothesis-test.ctm"
        os.system(f"cp {path} evaluation/{fileName}")

        mode = 'test'
        ret = evaluate2(
            mode=mode, output_file=fileName,
            evaluate_prefix='phoenix2014-groundtruth',
        )

        os.system(f"rm -rf evaluation/{fileName}")

if __name__ == '__main__':
    try:
        inputArgv = sys.argv[1]

        mode = inputArgv
    except:
        mode = 'test'

    evaluteMode(mode)
