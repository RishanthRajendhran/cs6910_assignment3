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
    inputChars.add("<UNKT>")    #For handling unseen tokens during test time
    targetChars.add("<UNKT>")   #For handling unseen tokens during test time

    inputChars = sorted(inputChars)
    targetChars = sorted(targetChars)
    numEncoderTokens = len(inputChars)
    numDecoderTokens = len(targetChars)
    maxEncoderSeqLen = max([len(txt) for txt in inputTexts])
    maxDecoderSeqLen = max([len(txt) for txt in targetTexts])

    print("Number of samples:", len(inputTexts))
    print("Number of unique input tokens:", numEncoderTokens)
    print("Number of unique output tokens:", numDecoderTokens)
    print("Max sequence length for inputs:", maxEncoderSeqLen)
    print("Max sequence length for outputs:", maxDecoderSeqLen)

    inputTokenIndex = dict([(char, i) for i, char in enumerate(inputChars)])
    targetTokenIndex = dict([(char, i) for i, char in enumerate(targetChars)])

    enocoderInputData = np.zeros(
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
            enocoderInputData[i, t, inputTokenIndex[char]] = 1.0
        enocoderInputData[i, t + 1 :, inputTokenIndex[" "]] = 1.0
        for t, char in enumerate(targetTxt):
            decoderInputData[i, t, targetTokenIndex[char]] = 1.0
            if t > 0:
                decoderTargetData[i, t - 1, targetTokenIndex[char]] = 1.0
        decoderInputData[i, t + 1 :, targetTokenIndex[" "]] = 1.0
        decoderTargetData[i, t:, targetTokenIndex[" "]] = 1.0

#Keras embedding layer works with indices and not one-hot encodings
enocoderInputData = np.argmax(enocoderInputData, axis=2)
decoderInputData = np.argmax(decoderInputData, axis=2)

#If best_model_acc has already been generated, the "-loadModel" flag can be used to load the model "./best_model_acc"
addAttention = config.addAttention
loadModel = False
if len(sys.argv) > 1:
    if "-addAttention" in sys.argv:
        addAttention = True
    if "-loadModel" in sys.argv:
        loadModel = True

s = seq2seq(
    numEncoderTokens, 
    numDecoderTokens, 
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

s.buildModel(
    enocoderInputData, 
    decoderInputData, 
    decoderTargetData
)

run.finish()

