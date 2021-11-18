import os
import pyedflib.highlevel
import random

"""读取CHB-MIT中的TXT和EDF文件"""

path = "chb-mit-scalp-eeg-database-1.0.0"  # 相对路径

edfFiles = []
txtFiles = []
# r = root, d = directories, f = files
for r, d, f in os.walk(path):  # 遍历path路径下的文件，将edf后缀的文件名保存在edfFiles里，txt后缀保存在txtFiles里
    for file in f:
        if file[-4:] == '.edf':
            edfFiles.append(os.path.join(r, file))
        elif file[-4:] == '.txt':
            txtFiles.append(os.path.join(r, file))

edfFiles = sorted(edfFiles)
txtFiles = sorted(txtFiles)

# for f in edfFiles:
#     print(f)
#
# for f in txtFiles:
#     print(f)


"""对EDF文件进行预处理--只保存23channels，去除多余channel"""


def DropChannels(edf):
    edf = 'chb-mit-scalp-eeg-database-1.0.0/chb20/chb20_11.edf'  # \chb01~\chb24
    # print(signals, '\n', signal_headers, '\n', header, '\n')
    edf_dropped = pyedflib.highlevel.drop_channels(edf, to_keep=['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3',
                                                                 'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4',
                                                                 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
                                                                 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8'])
    # signal是edf电平数据，signal_headers是所有channel的具体采样数据，header是该文件的病人、采样设备等数据
    signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_dropped)

    return signals


"""随机打乱edfFiles文件顺序"""

totalData = len(edfFiles)
random.shuffle(edfFiles)
partition = int(len(edfFiles) * 2 / 3)  # 打乱后的前2/3作训练集，后1/3作测试集
edfFilesVal = edfFiles[partition:]
edfFilesTrain = edfFiles[:partition]
trainData = len(edfFilesTrain)
valData = len(edfFilesVal)

# print(totalData, trainData, valData)

# for f in edfFilesTrain:
#     print(f)
#
# for f in edfFilesVal:
#     print(f)

