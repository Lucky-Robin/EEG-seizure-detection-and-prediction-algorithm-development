import os
import random

"""读取CHB-MIT中的TXT和EDF文件"""

path = "chb-mit-scalp-eeg-database-1.0.0"   # 相对路径

edfFiles = []
txtFiles = []
# r = root, d = directories, f = files
for r, d, f in os.walk(path):   # 遍历path路径下的文件，将edf后缀的文件名保存在edfFiles里，txt后缀保存在txtFiles里
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


"""对EDF文件进行预处理--未写"""


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

