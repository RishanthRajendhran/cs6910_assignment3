#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import keras as ke
import wandb
from wandb.keras import WandbCallback
import pandas as pd
import difflib 
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Layer as kerasLayers
import json

class AttentionLayer(kerasLayers):  #Bahudanau et al. attention model
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.w_a = self.add_weight(
                        shape=(input_shape[0][2],input_shape[0][2]), 
                        initializer="normal",
                        trainable=True,
                        name="w_a"
                    )
        self.u_a = self.add_weight(
                        shape=(input_shape[1][2],input_shape[1][2]), 
                        initializer="normal",
                        trainable=True,
                        name="u_a"
                    )
        self.v_a = self.add_weight(
                        shape=(input_shape[0][2],1), 
                        initializer="normal",
                        trainable=True,
                        name="v_a"
                    )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        decoderOuputs, encoderOutputs = x 

        stateC = ke.backend.sum(encoderOutputs, axis=1)
        stateE = ke.backend.sum(encoderOutputs, axis=2)

        def energyStep(inputs, states):
            eMaxSeqLen = encoderOutputs.shape[1]
            eHiddenStates = encoderOutputs.shape[2]

            w_a_dot_s = ke.backend.dot(encoderOutputs, self.w_a)

            u_a_dot_h = ke.backend.expand_dims(ke.backend.dot(inputs, self.u_a), 1)

            ws_plus_uh = ke.backend.tanh(w_a_dot_s + u_a_dot_h)

            e_i = ke.backend.squeeze(ke.backend.dot(ws_plus_uh, self.v_a), axis=-1)

            e_i = ke.backend.softmax(e_i)

            return e_i, [e_i]

        def contextStep(inputs, states):
            c_i = ke.backend.sum(encoderOutputs * ke.backend.expand_dims(inputs, -1), axis=1)

            return c_i, [c_i]

        lastOut, eOuts, _ = ke.backend.rnn(
            energyStep, decoderOuputs, [stateE]
        )

        lastOut, cOuts, _ = ke.backend.rnn(
            contextStep, eOuts, [stateC] 
        )

        return cOuts, eOuts

class seq2seq:
    def __init__(self, eNumTokens, dNumTokens, inputEmbedDim, cellType, eNumLayers, dNumLayers, hiddenStates, dropout, addAttention = False, loadModel = False, activationFn = "tanh", recurrentActivationFn = "sigmoid", batchSize = 64, epochs = 10, validationSplit = 0.2):
        self.eNumTokens = eNumTokens 
        self.dNumTokens = dNumTokens 
        self.inputEmbedDim = inputEmbedDim
        self.eType = cellType
        self.eNumLayers = eNumLayers
        self.eHiddenStates = hiddenStates
        self.dType = cellType
        self.dNumLayers = dNumLayers
        self.dHiddenStates = hiddenStates
        self.dropout = dropout
        self.addAttention = addAttention
        self.loadModel = loadModel
        self.activationFn = activationFn
        self.recurrentActivationFn = recurrentActivationFn
        self.batchSize = batchSize
        self.epochs = epochs
        self.validationSplit = validationSplit
        self.encoder_model = None
        self.decoder_model = None
        if  addAttention:           #Only supporting single layer encoder-decoder incase of attention requirement 
            self.eNumLayers = 1 
            self.dNumLayers = 1
            self.predFilePath = "./predictions_attention.txt"
        else: 
            self.predFilePath = "./predictions_vanilla.txt"


    def buildModel(self, encoderInput, decoderInput, decoderTarget):
        encoderInputs = ke.layers.Input(shape=(None,))
        encoderEmbedding = ke.layers.Embedding(input_dim=self.eNumTokens+1, output_dim=self.inputEmbedDim)(
            encoderInputs
        )
        stateHs, stateCs = [], []
        if self.eType == "lstm":
            encoderOutputs, stateH, stateC = ke.layers.LSTM(self.eHiddenStates, return_state=True, return_sequences=True, dropout = self.dropout, activation=self.activationFn, recurrent_activation=self.recurrentActivationFn)(
                encoderEmbedding
            )
            stateHs.append(stateH)
            stateCs.append(stateC)
            for i in range(1, self.eNumLayers):
                encoderOutputs, stateH, stateC = ke.layers.LSTM(self.eHiddenStates, return_state=True, return_sequences=True, dropout = self.dropout, activation=self.activationFn, recurrent_activation=self.recurrentActivationFn)(
                    encoderOutputs
                )
                stateHs.append(stateH)
                stateCs.append(stateC)
        elif self.eType == "gru":
            encoderOutputs, state = ke.layers.GRU(self.eHiddenStates, return_state=True, return_sequences=True, dropout = self.dropout, activation=self.activationFn)(
                encoderEmbedding
            )
            stateHs.append(state)
            for i in range(1, self.eNumLayers):
                encoderOutputs, state = ke.layers.GRU(self.eHiddenStates, return_state=True, return_sequences=True, dropout = self.dropout, activation=self.activationFn)(
                    encoderOutputs
                )  
                stateHs.append(state)
        elif self.eType == "rnn":      
            encoderOutputs, state = ke.layers.SimpleRNN(self.eHiddenStates, return_state=True, return_sequences=True, dropout = self.dropout, activation=self.activationFn)(
                encoderEmbedding
            )
            stateHs.append(state)
            for i in range(1, self.eNumLayers):
                encoderOutputs, state = ke.layers.SimpleRNN(self.eHiddenStates, return_state=True, return_sequences=True, dropout = self.dropout, activation=self.activationFn)(
                    encoderOutputs
                )  
                stateHs.append(state)
        else:
            print(f"Invalid encoder cell {self.eType}")
            exit(0)

        decoderInputs = ke.layers.Input(shape=(None,))
        decoderEmbedding = ke.layers.Embedding(input_dim=self.dNumTokens+1, output_dim=self.inputEmbedDim)(
            decoderInputs
        )
        if self.dType == "lstm":
            decoderOutputs, stateH, stateC = ke.layers.LSTM(self.dHiddenStates, return_sequences=True, return_state=True, dropout = self.dropout, activation=self.activationFn, recurrent_activation=self.recurrentActivationFn)(
                decoderEmbedding, initial_state=[stateHs[0], stateCs[0]]
            )
            for i in range(1, self.dNumLayers):
                decoderOutputs, stateH, stateC = ke.layers.LSTM(self.eHiddenStates, return_state=True, return_sequences=True, dropout = self.dropout, activation=self.activationFn, recurrent_activation=self.recurrentActivationFn)(
                    decoderOutputs, initial_state=[stateHs[i], stateCs[i]]
                )
        elif self.dType == "gru":
            decoderOutputs, state = ke.layers.GRU(self.dHiddenStates, return_sequences=True, return_state=True, dropout = self.dropout, activation=self.activationFn)(
                decoderEmbedding, initial_state=[stateHs[0]]
            )
            for i in range(1, self.dNumLayers):
                decoderOutputs, stateH = ke.layers.GRU(self.eHiddenStates, return_state=True, return_sequences=True, dropout = self.dropout, activation=self.activationFn)(
                    decoderOutputs, initial_state=[stateHs[i]]
                )  
        elif self.dType == "rnn":
            decoderOutputs, state = ke.layers.SimpleRNN(self.dHiddenStates, return_sequences=True, return_state=True, dropout = self.dropout, activation=self.activationFn)(
                decoderEmbedding, initial_state=[stateHs[0]]
            )
            for i in range(1, self.dNumLayers):
                decoderOutputs, stateH = ke.layers.SimpleRNN(self.eHiddenStates, return_state=True, return_sequences=True, dropout = self.dropout, activation=self.activationFn)(
                    decoderOutputs, initial_state=[stateHs[i]]
                )  
        else:
            print(f"Invalid decoder cell {self.eType}")
            exit(0)
        if self.addAttention:
            # attnOutputs = ke.layers.Attention(name='attention_vec')([decoderOutputs, encoderOutputs])
            attnOutputs, attnStates = AttentionLayer()([decoderOutputs, encoderOutputs])
            newDecoderOut = ke.layers.Concatenate(axis=-1)(
                [decoderOutputs, attnOutputs]
            )
        decoderDense = ke.layers.Dense(self.dNumTokens, activation="softmax")
        if self.addAttention:
            decoderOutputs = ke.layers.TimeDistributed(decoderDense, name='decoder_vec')(
                newDecoderOut
            )
        else:
            decoderOutputs = decoderDense(
                decoderOutputs
            )

        if self.addAttention:
            self.model = ke.Model([encoderInputs, decoderInputs], [decoderOutputs, attnOutputs])
        else:
            self.model = ke.Model([encoderInputs, decoderInputs], decoderOutputs)

        print(self.model.summary())

        if self.addAttention:
            filepath = './best_attn_model_acc'
            filepath_acc = './best_attn_model_acc'
            filepath_loss = './best_attn_model_loss'
            checkpoint_acc = ke.callbacks.ModelCheckpoint(filepath_acc, save_weights_only=False, name="val_categorical_accuracy", monitor='val_decoder_vec_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
            checkpoint_loss = ke.callbacks.ModelCheckpoint(filepath_loss, save_weights_only=False, name="val_loss", monitor='val_decoder_vec_loss', verbose=1, save_best_only=True, mode='min')
        else:
            filepath = './best_model_acc'
            filepath_acc = './best_model_acc'
            filepath_loss = './best_model_loss'
            checkpoint_acc = ke.callbacks.ModelCheckpoint(filepath_acc, save_weights_only=False, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
            checkpoint_loss = ke.callbacks.ModelCheckpoint(filepath_loss, save_weights_only=False, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        
        #Loading previously saved model - Onus on person running the program to ensure that the structure of the same model
        #is the same as the model built now
        if self.loadModel:
            self.model = ke.models.load_model(filepath, compile = True)

        callbacks_list = [WandbCallback(), checkpoint_acc, checkpoint_loss]

        if self.addAttention:
            self.model.compile(
                optimizer="rmsprop", 
                loss={"decoder_vec": ke.losses.CategoricalCrossentropy()},
                metrics={"decoder_vec": ke.metrics.CategoricalAccuracy()},
            )
            self.model.fit(
                [encoderInput, decoderInput],
                [decoderTarget,decoderTarget],
                batch_size = self.batchSize,
                epochs = self.epochs, 
                validation_split = self.validationSplit,
                callbacks=callbacks_list     
            )
        else:
            self.model.compile(
                optimizer="rmsprop", 
                loss=ke.losses.CategoricalCrossentropy(),
                metrics=[ke.metrics.CategoricalAccuracy()],
            )
            self.model.fit(
                [encoderInput, decoderInput],
                decoderTarget,
                batch_size = self.batchSize,
                epochs = self.epochs, 
                validation_split = self.validationSplit,
                callbacks=callbacks_list     
            )
        if not self.loadModel:
            ke.models.save_model(self.model, filepath)

    def defineEncoderDecoderModel(self, max_encoder_seq_length, forVisualisation=False, forTextVisualisation=False): #Hard coded for the best model obtained using sweep
        if self.loadModel == False:
            print("Compile and generate model first!")
            exit(0)
            
        #Loading previously saved model - Onus on person running the program to ensure that the structure of the same model
        #is the same as the model built now
        if self.addAttention:
            filepath = './best_attn_model_acc'
        else:
            filepath = './best_model_acc'

        self.model = ke.models.load_model(filepath, compile = True)
        for i in range(len(self.model.layers)):
            print(f"Layer {i} : {self.model.layers[i].name} == {self.model.layers[i]}")
        if not self.addAttention:
            encoder_inputs = self.model.input[0]
            encoder_states = []
            encoder_outputs, stateH = self.model.layers[4].output
            encoder_states.append(stateH)
            encoder_outputs, stateH = self.model.layers[6].output
            encoder_states.append(stateH)
            self.encoder_model = ke.Model(encoder_inputs, encoder_states)

            decoder_inputs = self.model.input[1]
            decoder_embedding = self.model.layers[3](
                decoder_inputs
            )
            decoder_stateH1 = ke.layers.Input(shape=(self.dHiddenStates,),name="inference_decoder_input_1")
            decoder_stateH2 = ke.layers.Input(shape=(self.dHiddenStates,),name="inference_decoder_input_2")
            decoder_states_inputs = [decoder_stateH1, decoder_stateH2]
            decoder_outputs, stateH1 = self.model.layers[5](
                decoder_embedding, initial_state=decoder_states_inputs[0:1]
            )
            decoder_outputs, stateH2 = self.model.layers[7](
                decoder_outputs, initial_state=decoder_states_inputs[1:2]
            )
            decoder_states = [stateH1, stateH2]

            decoder_dense = self.model.layers[8]    
            decoder_final_outputs = decoder_dense(decoder_outputs)
            if forVisualisation or forTextVisualisation:
                self.decoder_model = ke.Model(
                    [decoder_inputs] + decoder_states_inputs, [decoder_final_outputs] + [decoder_outputs] + decoder_states
                )
            else:
                self.decoder_model = ke.Model(
                    [decoder_inputs] + decoder_states_inputs, [decoder_final_outputs] + decoder_states
                )
        else:
            encoder_inputs = self.model.input[0]
            encoder_states = []
            encoder_outputs, stateH, stateC = self.model.layers[4].output
            encoder_states = [stateH, stateC]
            self.encoder_model = ke.Model(encoder_inputs, [encoder_outputs, encoder_states])

            decoder_inputs = self.model.input[1]
            decoder_embedding = self.model.layers[3](
                decoder_inputs
            )
            decoder_stateH = ke.layers.Input(shape=(self.dHiddenStates,),name="inference_decoder_input_stateH")
            decoder_stateC = ke.layers.Input(shape=(self.dHiddenStates,),name="inference_decoder_input_stateC")
            decoder_states_inputs = [decoder_stateH, decoder_stateC]
            decoder_outputs, stateH, stateC = self.model.layers[5](
                decoder_embedding, initial_state=decoder_states_inputs
            )
            decoder_states = [stateH, stateC]
            decoder_encoder_outputs = ke.layers.Input(shape=(max_encoder_seq_length,self.dHiddenStates),name="inference_encoder_output")
            attention_outputs, attention_states = self.model.layers[6]([decoder_outputs, decoder_encoder_outputs])
            concatenated_outputs = self.model.layers[7]([decoder_outputs, attention_outputs])

            decoder_dense = self.model.layers[8]    
            decoder_final_outputs = decoder_dense(concatenated_outputs)
            if forTextVisualisation:
                self.decoder_model = ke.Model(
                    [decoder_inputs] + [decoder_encoder_outputs] + decoder_states_inputs, [decoder_final_outputs] + [attention_states] + [decoder_outputs] + decoder_states
                )
            elif forVisualisation:
                self.decoder_model = ke.Model(
                    [decoder_inputs] + [decoder_encoder_outputs] + decoder_states_inputs, [decoder_final_outputs] + [attention_states] + decoder_states
                )
            else:
                self.decoder_model = ke.Model(
                    [decoder_inputs] + [decoder_encoder_outputs] + decoder_states_inputs, [decoder_final_outputs] + decoder_states
                )

    def testModel(self, inputTexts, outputTexts, inputData, outputData, input_token_index, target_token_index, max_encoder_seq_length, max_decoder_seq_length, puncPos=[], nextLinePos=[]):

        self.defineEncoderDecoderModel(max_encoder_seq_length)

        acc = 0
        #If the number of corrections to be made to the decoder output to get the correct output sequence is less than 3
        #provided the correct decoder output sequence and the decoder output is at least twice as long as the number of corrections to be made
        #we consider it a partial match
        partialMatch = 0    #If the prediction was reasonably close
        countWord = 0
        with open(self.predFilePath,"w") as predFile:
            for seq_index in range(len(inputData)):
                input_seq = inputData[seq_index : seq_index + 1]
                output_seq = outputData[seq_index : seq_index + 1]
                decoded_sentence = self.decodeSequence(input_seq, inputData, outputData, input_token_index, target_token_index, max_encoder_seq_length, max_decoder_seq_length)
                if decoded_sentence[-1] == "\n":
                    predFile.write(decoded_sentence[:-1])
                else: 
                    predFile.write(decoded_sentence)
                if countWord in nextLinePos:
                    predFile.write("\n")
                    predFile.flush()  #continuous file i/o slows down execution - use only for debugging purpose
                elif countWord in puncPos.keys():
                    predFile.write(puncPos[countWord])
                else: 
                    predFile.write(" ")
                countWord += 1
                print("--------------------------------------")
                print("Input sequence:",inputTexts[seq_index])
                print("Output sequence:",outputTexts[seq_index])
                print("Decoded sentence: ", decoded_sentence)
                isCorrect = (outputTexts[seq_index].strip("\t").lower() == decoded_sentence.strip("\t").lower())
                print("Correct Prediction?: "+str(isCorrect))
                numCorr = sum([0 if s[0] == " " else 1 for (i,s) in list(enumerate(difflib.ndiff(decoded_sentence.strip("\t").lower(),outputTexts[seq_index].strip("\t").lower())))])
                if numCorr:
                    print(f"Number of corrections required: {numCorr}")
                    if numCorr <=2 and len(outputTexts[seq_index]) >= max(5,2*numCorr) and len(decoded_sentence) >= max(5,2*numCorr):
                        partialMatch += 1
                else: 
                    print("No corrections required!")
                acc += isCorrect
                print()
                print(f"Number of words decoded until now: {seq_index}")
                print(f"Number of perfect matches until now: {acc}")
                print(f"Number of partial matches until now: {partialMatch}")
                print("--------------------------------------")
            acc /= len(inputData)
            print("Word-level accuracy: "+ str(acc))


    def decodeSequence(self, input_seq, inputData, outputData, input_token_index, target_token_index, max_encoder_seq_length, max_decoder_seq_length):
        reverse_input_char_index = dict((i,char) for char, i in input_token_index.items())
        reverse_target_char_index = dict((i,char) for char, i in target_token_index.items())

        if not self.addAttention:
            states_value = [self.encoder_model.predict(input_seq)]
        else:
            encoder_outputs, states_value = self.encoder_model.predict(input_seq)
            # encoder_outputs = np.argmax(encoder_outputs,axis=1)

        target_seq = np.zeros((1,1,self.dNumTokens))
        target_seq[0, 0, target_token_index["\t"]] = 1.0 

        stop_condition = False 
        decoded_sentence = ""
        i = 0
        while not stop_condition:
            inp_target_seq = np.argmax(target_seq,axis=2)
            if not self.addAttention:
                output_token, s1, s2 = self.decoder_model.predict([inp_target_seq] + states_value)  
            else:
                output_token, sH, sC = self.decoder_model.predict([inp_target_seq] + [encoder_outputs] + states_value)
            i += 1 
            sampled_token_index = np.argmax(output_token[0, 0])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True 
            
            target_seq = np.zeros((1, 1, self.dNumTokens))
            target_seq[0, 0, sampled_token_index] = 1.0
            
            if not self.addAttention:
                states_value = [s1, s2]
            else:
                states_value = [sH, sC]

        return decoded_sentence

    def visualiseAttention(self, inputTexts, outputTexts, inputData, outputData, input_token_index, target_token_index, max_encoder_seq_length, max_decoder_seq_length):
        
        #Plots attention heatmaps for 10 words from the test dataset

        if self.loadModel == False:
            print("Compile and generate model first!")
            return 
        if self.addAttention == False:
            print("Set addAttention to True first!")
            return 

        self.defineEncoderDecoderModel(max_encoder_seq_length, True)

        reverse_input_char_index = dict((i,char) for char, i in input_token_index.items())
        reverse_target_char_index = dict((i,char) for char, i in target_token_index.items())
        matplotlib.rc("font", family="Tamil Sangam MN")
 
        # maxPlots = len(inputData)
        maxPlots = 10
        start = np.random.randint(0, len(inputData)-maxPlots)
        for seq_index in range(start, start + maxPlots):
            input_seq = inputData[seq_index : seq_index + 1]

            encoder_outputs, states_value = self.encoder_model.predict(input_seq)

            target_seq = np.zeros((1,1,self.dNumTokens))
            target_seq[0, 0, target_token_index["\t"]] = 1.0 

            stop_condition = False 
            decoded_sentence = ""
            visualiseOuts = []
            i = 0
            while not stop_condition:
                inp_target_seq = np.argmax(target_seq,axis=2)
                output_token, attnStates, sH, sC = self.decoder_model.predict([inp_target_seq] + [encoder_outputs] + states_value)
                i += 1 
                visualiseOuts.append((np.argmax(output_token, axis=-1)[0, 0], attnStates))
                sampled_token_index = np.argmax(output_token[0, 0])
                sampled_char = reverse_target_char_index[sampled_token_index]
                decoded_sentence += sampled_char

                if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                    stop_condition = True 
                
                target_seq = np.zeros((1, 1, self.dNumTokens))
                target_seq[0, 0, sampled_token_index] = 1.0
                
                states_value = [sH, sC]

            # print("--------------------------------------")
            # print("Input sequence:",inputTexts[seq_index])
            # print("Output sequence:",outputTexts[seq_index])
            # print("Decoded sentence: ", decoded_sentence)

            attn, dec = [], []
            for dec_i, attn_i in visualiseOuts:
                attn.append(attn_i.reshape(-1))
                dec.append(dec_i)
            visualiseOuts = np.transpose(np.array(attn,dtype=float))
            x = [reverse_target_char_index[token] for token in dec]
            y = [token for token in inputTexts[seq_index]]
            plt.imshow(visualiseOuts[:len(y),:])
            plt.xticks(ticks=np.arange(len(x)), labels=x)
            plt.yticks(ticks=np.arange(len(y)), labels=y)
            plt.title(f"Input sequence: {inputTexts[seq_index]} | Output sequence: {outputTexts[seq_index]} | Decoded sentence: {decoded_sentence}")
            wandb.log({f"plot_{seq_index-start}": wandb.Image(plt)})

    def visualiseTextAttention(self, inputTexts, outputTexts, inputData, outputData, input_token_index, target_token_index, max_encoder_seq_length, max_decoder_seq_length, textConfidence=False):
        
        #Plots text attention heatmaps for 10 words from the test dataset

        if self.loadModel == False:
            print("Compile and generate model first!")
            return 

        self.defineEncoderDecoderModel(max_encoder_seq_length, True, True)

        reverse_input_char_index = dict((i,char) for char, i in input_token_index.items())
        reverse_target_char_index = dict((i,char) for char, i in target_token_index.items())
        matplotlib.rc("font", family="Tamil Sangam MN")
 
        # maxPlots = len(inputData)
        maxPlots = 10
        start = np.random.randint(0, len(inputData)-maxPlots)
        words = []
        predsList = []
        attentionsList = []
        predictions = []
        allOuts = []
        allAtts = []
        for seq_index in range(start, start + maxPlots):
            input_seq = inputData[seq_index : seq_index + 1]

            if self.addAttention:
                encoder_outputs, states_value = self.encoder_model.predict(input_seq)
            else: 
                states_value = self.encoder_model.predict(input_seq)

            target_seq = np.zeros((1,1,self.dNumTokens))
            target_seq[0, 0, target_token_index["\t"]] = 1.0 

            stop_condition = False 
            decoded_sentence = ""
            i = 0
            attnsList = []
            maxOutsList = []
    
            while not stop_condition:
                inp_target_seq = np.argmax(target_seq,axis=2)
                if self.addAttention:
                    output_token, attnStates, lastDecOuts, sH, sC = self.decoder_model.predict([inp_target_seq] + [encoder_outputs] + states_value)
                    maxOuts = output_token[0, 0]
                    s1 = np.argsort(maxOuts)[-1]
                    s2 = np.argsort(maxOuts)[-2]
                    s3 = np.argsort(maxOuts)[-3]
                    s1 = reverse_target_char_index[s1]
                    s2 = reverse_target_char_index[s2]
                    s3 = reverse_target_char_index[s3]
                    attnsList.append(attnStates.reshape(-1).tolist()[:len(inputTexts[seq_index])])
                    s1 = (s1).strip()
                    s2 = (s2).strip()
                    s3 = (s3).strip()
                    maxOutsList.append([s1, s2, s3])
                    allAtts.append(lastDecOuts.reshape(-1).tolist())
                    allOuts.append(reverse_target_char_index[np.argmax(output_token[0, 0])])
                else: 
                    output_token, attnStates, sH, sC = self.decoder_model.predict([inp_target_seq] + states_value)
                    # attnStates = (1/(1 + np.exp(-attnStates)))
                    allAtts.append(attnStates.reshape(-1).tolist())
                    allOuts.append(reverse_target_char_index[np.argmax(output_token[0, 0])])
                i += 1 
                sampled_token_index = np.argmax(output_token[0, 0])
                sampled_char = reverse_target_char_index[sampled_token_index]
                decoded_sentence += sampled_char

                if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                    stop_condition = True 
                
                target_seq = np.zeros((1, 1, self.dNumTokens))
                target_seq[0, 0, sampled_token_index] = 1.0
                
                states_value = [sH, sC]
            print("--------------------------------------")
            print("Input sequence:",inputTexts[seq_index])
            print("Output sequence:",outputTexts[seq_index])
            print("Decoded sentence: ", decoded_sentence)
            words.append(inputTexts[seq_index])
            predsList.append(maxOutsList)
            attentionsList.append(attnsList)
            predictions.append(decoded_sentence.strip())
        if self.addAttention and not textConfidence:
            lines = []
            with open("./textAttentionTemplate.html", "r") as f: 
                lines = f.readlines()
            
            line63 = [f"\t\twords = {json.dumps(words)};\n"]
            line64 = [f"\t\tpredsList = {json.dumps(predsList)};\n"]
            line65 = [f"\t\tattentionsList = {json.dumps(attentionsList)};\n"]
            line66 = [f"\t\tpredictions = {json.dumps(predictions)};\n"]
            lines = lines[:63] + line63 + line64 + line65 + line66 + lines[63:]
            with open("./textAttentionResult.html", "w") as f: 
                f.writelines(lines)
        else: 
            lines = []
            with open("./textConfidenceTemplate.html", "r") as f: 
                lines = f.readlines()

            allAtts = np.transpose(allAtts)
            allAtts = (allAtts - np.mean(allAtts, axis=0))
            allAtts = 1/(1+np.exp(-allAtts))
            allAtts = allAtts.tolist()
            
            line47 = [f"\t\tpredsList = {json.dumps(allOuts)};\n"]
            line48 = [f"\t\tattnList = {json.dumps(allAtts)};\n"]
            lines = lines[:47] + line47 + line48 + lines[47:]
            with open("./textConfidenceResult.html", "w") as f: 
                f.writelines(lines)

    def predictRest(self, target_seq, encoder_outputs, states_value, reverse_target_char_index, decoded_sentence, max_decoder_seq_length):
        inp_target_seq = np.argmax(target_seq,axis=2)
        output_token, attnStates, sH, sC = self.decoder_model.predict([inp_target_seq] + [encoder_outputs] + states_value)
        sampled_token_index = np.argmax(output_token[0, 0])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            return decoded_sentence
        
        target_seq = np.zeros((1, 1, self.dNumTokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        
        states_value = [sH, sC]

        return self.predictRest(target_seq, encoder_outputs, states_value, reverse_target_char_index, decoded_sentence, max_decoder_seq_length)

    