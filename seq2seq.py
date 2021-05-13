import numpy as np
import keras as ke
from wandb.keras import WandbCallback
import pandas as pd
import difflib 

class CharVal(object):
    def __init__(self, ch, vl):
        self.char = ch 
        self.val = vl 

    def __str__(self):
        return self.char 

def rgbToHex(rgb):
    return "#%02x%02x%02x" % rgb 

def colorCharVals(s):
    r = 255 - int(s.val*255)
    colour = rgbToHex((255, r,r))
    return "background-color: %s" % colour

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
            attnOutputs = ke.layers.Attention(name='attention_vec')([decoderOutputs, encoderOutputs])
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

        #Loading previously saved model - Onus on person running the program to ensure that the structure of the same model
        #is the same as the model built now
        if self.loadModel:
            self.model = ke.models.load_model(filepath, compile = True)

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

    def testModel(self, inputTexts, outputTexts, inputData, outputData, input_token_index, target_token_index, max_encoder_seq_length, max_decoder_seq_length): #Hard coded for the best model obtained using sweep
        if self.loadModel == False:
            print("Compile and generate model first!")
            return 
            
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
            decoder_outputs = decoder_dense(decoder_outputs)
            self.decoder_model = ke.Model(
                [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
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
            attention_outputs = self.model.layers[6]([decoder_outputs, decoder_encoder_outputs])
            concatenated_outputs = self.model.layers[7]([decoder_outputs, attention_outputs])

            decoder_dense = self.model.layers[8]    
            decoder_outputs = decoder_dense(concatenated_outputs)
            self.decoder_model = ke.Model(
                [decoder_inputs] + [decoder_encoder_outputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
            )

        acc = 0
        #If the number of corrections to be made to the decoder output to get the correct output sequence is less than 3
        #provided the correct decoder output sequence and the decoder output is at least twice as long as the number of corrections to be made
        #we consider it a partial match
        partialMatch = 0    #If the prediction was reasonably close

        # enocoderOutputs, encoderStates = self.encoder_model.predict(inputData)
        # np.save("encoderOutputs.npy",enocoderOutputs)
        # decoderOutputs, decoderStates = self.decoder_model.predict(outputData + [enocoderOutputs] + encoderStates)
        # np.save("decoderOutputs.npy",decoderOutputs)

        for seq_index in range(len(inputData)):
            input_seq = inputData[seq_index : seq_index + 1]
            output_seq = outputData[seq_index : seq_index + 1]
            decoded_sentence = self.decodeSequence(input_seq, inputData, outputData, input_token_index, target_token_index, max_encoder_seq_length, max_decoder_seq_length)
            print("--------------------------------------")
            print("Input sequence:",inputTexts[seq_index])
            print("Output sequence:",outputTexts[seq_index])
            print("Decoded sentence: ", decoded_sentence)
            isCorrect = (outputTexts[seq_index].strip("\t") == decoded_sentence.strip("\t"))
            print("Correct Prediction?: "+str(isCorrect))
            numCorr = sum([0 if s[0] == " " else 1 for (i,s) in list(enumerate(difflib.ndiff(decoded_sentence.strip("\t"),outputTexts[seq_index].strip("\t"))))])
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

    def visualiseAttention(self, encoderData, decoderData, inputTokens, targetTokens):
        if self.loadModel == False:
            print("Compile and generate model first!")
            return 
        if self.addAttention == False:
            print("Set addAttention to True first!")
            return 

        #Loading previously saved model - Onus on person running the program to ensure that the structure of the same model
        #is the same as the model built now
        if self.addAttention:
            filepath = './best_attn_model_acc'
        else:
            filepath = './best_model_acc'
            
        self.model = ke.models.load_model(filepath, compile = True)
        for i in range(len(self.model.layers)):
            print(f"Layer {i} : {self.model.layers[i].name} == {self.model.layers[i]}")
        # outputs = self.model.predict([encoderData, decoderData])
        # decoderOutput, attentionOutputs = outputs[0], outputs[1]
        # print(self.model.layers[6].get_weights())
        # print(np.array(self.model.layers[6].get_weights()).shape)


        # np.save("decoderData.npy",decoderData)
        # print(decoderData.shape)
        # attentionOutputs = np.load("attnOuts.npy")
        # char_vals = [CharVal(c, v) for c, v in zip(inputTokens, np.argmax(attentionOutputs,axis=0))]
        # char_df = pd.DataFrame(char_vals).transpose()
        # char_df = char_df.style.applymap(colorCharVals)
        # # char_df.to_excel("char_df.xlsx")
        # print(char_df)
        # html = char_df.background_gradient().render()
        # print(html)
        # with open("./char_df.html","w") as fp:
        #     fp.write(html)

    