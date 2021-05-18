import numpy as np
from seq2seq import seq2seq
import sys

import wandb
#Best Model config from sweep
if "-addAttention" in sys.argv:
    hyperparameters_default = dict(
        inputEmbedDim = 64,
        numLayers = 2,
        hiddenStates = 64,
        cellType = "lstm",
        dropout = 0.3,
        activationFn = "tanh",
        recurrentActivationFn = "sigmoid",
        batchSize = 64,
        numEpochs = 10,
        validationSplit = 0.2,
        addAttention = True,
    )
else:
    hyperparameters_default = dict(
        inputEmbedDim = 256,
        numLayers = 2,
        hiddenStates = 32,
        cellType = "gru",
        dropout = 0.2,
        activationFn = "relu",
        recurrentActivationFn = "tanh",
        batchSize = 64,
        numEpochs = 10,
        validationSplit = 0.2,
        addAttention = False,
    )

run = wandb.init(project="cs6910_assignment3", entity="rishanthrajendhran", config=hyperparameters_default)
config = wandb.config
wandb.run.name = f"{config.inputEmbedDim}_{config.numLayers}_{config.hiddenStates}_{config.cellType}_{config.dropout}_{config.activationFn}_{config.recurrentActivationFn}"
wandb.run.save(wandb.run.name)

punctuations = ['\n', '\t', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

inputTexts = []
targetTexts = []
inputChars = set()
targetChars = set()
langCodes = ["ta"]
for langCode in langCodes:
    dataPath = f"./dakshina_dataset_v1.0/{langCode}/romanized/{langCode}.romanized.rejoined.aligned.cased_nopunct.tsv"
    with open(dataPath, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in lines:
        if line == "":
            continue
        splitWords = line.split("\t")
        inputText, targetText = splitWords[0], splitWords[1]
        targetText = "\t" + targetText + "\n"
        if inputText == "</s>":
            continue
        # print(f"{inputText} <=> {targetText}")
        inputTexts.append(inputText)
        targetTexts.append(targetText)
        for ch in inputText:
            if ch not in inputChars:
                inputChars.add(ch)
        for ch in targetText:
            if ch not in targetChars:
                targetChars.add(ch)

    inputChars.add("\t")
    targetChars.add("\t")
    inputChars.add("<UNKT>")
    targetChars.add("<UNKT>")
    original_testInputTexts = inputTexts
    original_testOutputTexts = targetTexts
    original_testInputTokens = list(inputChars)
    original_testTargetTokens = list(targetChars)

    inputChars = sorted(inputChars)
    targetChars = sorted(targetChars)
    original_numEncoderTokens = len(inputChars)
    original_numDecoderTokens = len(targetChars)
    original_maxEncoderSeqLen = max([len(txt) for txt in inputTexts])
    original_maxDecoderSeqLen = max([len(txt) for txt in targetTexts])

    print("Train data set:")
    print("Number of samples:", len(inputTexts))
    print("Number of unique input tokens:", original_numEncoderTokens)
    print("Number of unique output tokens:", original_numDecoderTokens)
    print("Max sequence length for inputs:", original_maxEncoderSeqLen)
    print("Max sequence length for outputs:", original_maxDecoderSeqLen)

    original_inputTokenIndex = dict([(char, i) for i, char in enumerate(inputChars)])
    original_targetTokenIndex = dict([(char, i) for i, char in enumerate(targetChars)])

    original_encoderInputData = np.zeros(
        (len(inputTexts), original_maxEncoderSeqLen, original_numEncoderTokens), dtype="float32"
    )
    original_decoderInputData = np.zeros(
        (len(inputTexts), original_maxDecoderSeqLen, original_numDecoderTokens), dtype="float32"
    )
    original_decoderTargetData = np.zeros(
        (len(inputTexts), original_maxDecoderSeqLen, original_numDecoderTokens), dtype="float32"
    )

    for i, (inputText, targetTxt) in enumerate(zip(inputTexts, targetTexts)):
        for t, char in enumerate(inputText):
            original_encoderInputData[i, t, original_inputTokenIndex[char]] = 1.0
        original_encoderInputData[i, t + 1 :, original_inputTokenIndex[" "]] = 1.0
        for t, char in enumerate(targetTxt):
            original_decoderInputData[i, t, original_targetTokenIndex[char]] = 1.0
            if t > 0:
                original_decoderTargetData[i, t - 1, original_targetTokenIndex[char]] = 1.0
        original_decoderInputData[i, t + 1 :, original_targetTokenIndex[" "]] = 1.0
        original_decoderTargetData[i, t:, original_targetTokenIndex[" "]] = 1.0

#Keras embedding layer works with indices and not one-hot encodings
original_encoderInputData = np.argmax(original_encoderInputData, axis=2)
original_decoderInputData = np.argmax(original_decoderInputData, axis=2)
#############################################################################################################################

inputTexts = []
targetTexts = []
inputChars = set()
targetChars = set()
langCodes = ["ta"]
puncPos = {}
nextLinePos = []
countWords = 0
for langCode in langCodes:
    nativeDataPath = f"./dakshina_dataset_v1.0/{langCode}/romanized/{langCode}.romanized.rejoined.test.native.txt"
    romanDataPath = f"./dakshina_dataset_v1.0/{langCode}/romanized/{langCode}.romanized.rejoined.test.roman.txt"
    nativeLines = []
    romanLines = []
    with open(nativeDataPath, "r", encoding="utf-8") as native:
        nativeLines = native.read().split("\n")
    with open(romanDataPath, "r", encoding="utf-8") as roman:
        romanLines = roman.read().split("\n")
    for (nativeLine, romanLine) in zip(nativeLines, romanLines):
        if nativeLine == "" or romanLine == "":
            continue
        splitNativeWords = nativeLine.split(" ")
        splitRomanWords = romanLine.split(" ")
        for (inputText, targetText) in zip(splitNativeWords, splitRomanWords):
            if inputText[-1] in punctuations:
                inputText = inputText[:-1]
            if targetText[-1] in punctuations:
                if targetText[-1] not in ["\n", "\t"]:
                    puncPos[countWords] = targetText[-1]
                targetText = targetText[:-1]
            # print(f"{inputText} <=> {targetText}")
            targetText = "\t" + targetText + "\n"
            inputTexts.append(inputText)
            targetTexts.append(targetText)
            countWords += 1
            for ch in inputText:
                if ch not in inputChars:
                    inputChars.add(ch)
            for ch in targetText:
                if ch not in targetChars:
                    targetChars.add(ch)
        nextLinePos.append(countWords-1)

    inputChars.add("\t")
    targetChars.add("\t")
    testInputTexts = inputTexts
    testOutputTexts = targetTexts

    inputChars = sorted(inputChars)
    targetChars = sorted(targetChars)
    numEncoderTokens = len(inputChars)
    numDecoderTokens = len(targetChars)
    maxEncoderSeqLen = max([len(txt) for txt in inputTexts])
    maxDecoderSeqLen = max([len(txt) for txt in targetTexts])

    print("Test data set:")
    print("Number of samples:", len(inputTexts))
    print("Number of unique input tokens:", numEncoderTokens)
    print("Number of unique output tokens:", numDecoderTokens)
    print("Max sequence length for inputs:", maxEncoderSeqLen)
    print("Max sequence length for outputs:", maxDecoderSeqLen)

    maxEncoderSeqLen = original_maxEncoderSeqLen
    maxDecoderSeqLen = original_maxDecoderSeqLen
    numEncoderTokens = original_numEncoderTokens
    numDecoderTokens = original_numDecoderTokens

    # inputTokenIndex = dict([(char, i) for i, char in enumerate(inputChars)])
    # targetTokenIndex = dict([(char, i) for i, char in enumerate(targetChars)])
    inputTokenIndex = original_inputTokenIndex
    targetTokenIndex = original_targetTokenIndex

    encoderInputData = np.zeros(
        (len(inputTexts), maxEncoderSeqLen, numEncoderTokens), dtype="float32"
    )
    decoderInputData = np.zeros(
        (len(inputTexts), maxDecoderSeqLen, numDecoderTokens), dtype="float32"
    )
    decoderTargetData = np.zeros(
        (len(inputTexts), maxDecoderSeqLen, numDecoderTokens), dtype="float32"
    )

    for i, (inputText, targetTxt) in enumerate(zip(inputTexts, targetTexts)):
        for t, char in enumerate(inputText):
            if char not in inputTokenIndex.keys():
                char = "<UNKT>"
            encoderInputData[i, t, inputTokenIndex[char]] = 1.0
        encoderInputData[i, t + 1 :, inputTokenIndex[" "]] = 1.0
        for t, char in enumerate(targetTxt):
            if char not in targetTokenIndex.keys():
                char = "<UNKT>"
            decoderInputData[i, t, targetTokenIndex[char]] = 1.0
            if t > 0:
                decoderTargetData[i, t - 1, targetTokenIndex[char]] = 1.0
        decoderInputData[i, t + 1 :, targetTokenIndex[" "]] = 1.0
        decoderTargetData[i, t:, targetTokenIndex[" "]] = 1.0

#Keras embedding layer works with indices and not one-hot encodings
encoderInputData = np.argmax(encoderInputData, axis=2)
decoderInputData = np.argmax(decoderInputData, axis=2)


#############################################################################################################################
#If best_model_acc has already been generated, the "-loadModel" flag can be used to load the model "./best_model_acc"
addAttention = config.addAttention
loadModel = False
if len(sys.argv) > 1:
    if "-addAttention" in sys.argv:
        addAttention = True
    if "-loadModel" in sys.argv:
        loadModel = True

print(f"Shape of original_encoder_data = {original_encoderInputData.shape}")
print(f"Shape of original_decoder_data = {original_decoderInputData.shape}")
print(f"Shape of encoder_data = {encoderInputData.shape}")
print(f"Shape of decoder_data = {decoderInputData.shape}")

s = seq2seq(
    original_numEncoderTokens, 
    original_numDecoderTokens, 
    config.inputEmbedDim, 
    config.cellType, 
    config.numLayers, 
    config.numLayers, 
    config.hiddenStates, 
    config.dropout, 
    addAttention,
    loadModel, 
    config.activationFn, 
    config.recurrentActivationFn,
    config.batchSize,
    config.numEpochs,
    config.validationSplit
)

if "-visualiseTextAttention" in sys.argv:
    s.visualiseTextAttention(
        testInputTexts,
        testOutputTexts,
        encoderInputData, 
        decoderInputData,
        original_inputTokenIndex,
        original_targetTokenIndex,
        maxEncoderSeqLen,
        maxDecoderSeqLen
    )
elif "-visualiseAttention" in sys.argv:
    s.visualiseAttention(
        testInputTexts,
        testOutputTexts,
        encoderInputData, 
        decoderInputData,
        original_inputTokenIndex,
        original_targetTokenIndex,
        maxEncoderSeqLen,
        maxDecoderSeqLen
    )
else:
    s.testModel(
        testInputTexts,
        testOutputTexts,
        encoderInputData, 
        decoderInputData,
        original_inputTokenIndex,
        original_targetTokenIndex,
        maxEncoderSeqLen,
        maxDecoderSeqLen,
        puncPos,    #For output purpose
        nextLinePos
    )
    

run.finish()

