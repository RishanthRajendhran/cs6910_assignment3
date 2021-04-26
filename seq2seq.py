import numpy as np
import keras as ke

class seq2seq:
    def __init__(self, eNumTokens, dNumTokens, inputEmbedDim, eType, eNumLayers, eHiddenStates, dType, dNumLayers, dHiddenStates, batchSize = 64, epochs = 10, validationSplit = 0.2):
        self.eNumTokens = eNumTokens
        self.dNumTokens = dNumTokens
        self.inputEmbedDim = inputEmbedDim
        self.eType = eType
        self.eNumLayers = eNumLayers
        self.eHiddenStates = eHiddenStates
        self.dType = dType
        self.dNumLayers = dNumLayers
        self.dHiddenStates = dHiddenStates
        self.batchSize = batchSize
        self.epochs = epochs
        self.validationSplit = validationSplit

    def buildModel(self, encoderInput, decoderInput, decoderTarget):
        eInp = ke.layers.Input(shape=(None,))
        encoderEmbedded = ke.layers.Embedding(input_dim= self.eNumTokens, output_dim=self.inputEmbedDim)(
            eInp
        )
        eOutput, eStateH, eStateC = ke.layers.LSTM(self.eHiddenStates, return_state=True, return_sequences=True)(
            encoderEmbedded
        )
        # for i in  range(1, self.eNumLayers):
        #     eOutput, eStateH, eStateC = ke.layers.LSTM(self.eHiddenStates, return_state=True)(
        #         eOutput, initial_state=[eStateH, eStateC]
        #     )

        dInp = ke.layers.Input(shape=(None,))
        decoderEmbedded = ke.layers.Embedding(input_dim= self.dNumTokens, output_dim=self.inputEmbedDim)(
            dInp
        )
        dOutput, dStateH, dStateC = ke.layers.LSTM(self.dHiddenStates, return_state=True)(
            decoderEmbedded, initial_state=[eStateH, eStateC]
        )
        # for i in  range(1, self.dNumLayers):
        #     dOutput, dStateH, dStateC = ke.layers.LSTM(self.dHiddenStates, return_state=True)(
        #         dOutput, initial_state=[dStateH, dStateC]
        #     )
        self.model = ke.Model([eInp, dInp], eOutput)
        # ke.layers.LSTM(self.eHiddenStates, return_state=True)
        # self.model = ke.Sequential()
        # self.model.add(ke.layers.Embedding(input_dim= self.vocabSize, output_dim=self.inputEmbedDim))
        # for i in  range(self.eNumLayers):
        #     if self.eType == "lstm":
        #         self.model.add(ke.layers.LSTM(self.eHiddenStates, return_sequences=True))
        # for i in range(self.dNumLayers):
        #     if self.dType == "lstm":
        #         self.model.add(ke.layers.LSTM(self.dHiddenStates, return_sequences=True))
        print(self.model.summary())
        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics = ["accuracy"])
        self.model.fit(
            [encoderInput, decoderInput],
            decoderTarget,
            batch_size = self.batchSize,
            epochs = self.epochs, 
            validation_split = self.validationSplit      
        )