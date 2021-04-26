import numpy as np
import keras
from seq2seq import seq2seq

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

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
        if inputText == "</s>":
            continue
        print(f"{inputText} <=> {targetText}")
        inputTexts.append(inputText)
        targetTexts.append(targetText)
        for ch in inputText:
            if ch not in inputChars:
                inputChars.add(ch)
        for ch in targetText:
            if ch not in targetChars:
                targetChars.add(ch)

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

    input_token_index = dict([(char, i) for i, char in enumerate(inputChars)])
    target_token_index = dict([(char, i) for i, char in enumerate(targetChars)])

    encoder_input_data = np.zeros(
        (len(inputTexts), maxEncoderSeqLen, numEncoderTokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(inputTexts), maxDecoderSeqLen, numDecoderTokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(inputTexts), maxDecoderSeqLen, numDecoderTokens), dtype="float32"
    )

    for i, (input_text, target_text) in enumerate(zip(inputTexts, targetTexts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
        decoder_target_data[i, t:, target_token_index[" "]] = 1.0

    # # Define an input sequence and process it.
    # encoder_inputs = keras.Input(shape=(None, numEncoderTokens))
    # encoder = keras.layers.LSTM(latent_dim, return_state=True)
    # encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # # We discard `encoder_outputs` and only keep the states.
    # encoder_states = [state_h, state_c]

    # # Set up the decoder, using `encoder_states` as initial state.
    # decoder_inputs = keras.Input(shape=(None, numDecoderTokens))

    # # We set up our decoder to return full output sequences,
    # # and to return internal states as well. We don't use the
    # # return states in the training model, but we will use them in inference.
    # decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    # decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    # decoder_dense = keras.layers.Dense(numDecoderTokens, activation="softmax")
    # decoder_outputs = decoder_dense(decoder_outputs)

    # # Define the model that will turn
    # # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    # model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # model.compile(
    #     optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    # )
    # model.fit(
    #     [encoder_input_data, decoder_input_data],
    #     decoder_target_data,
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     validation_split=0.2,
    # )
    # # Save model
    # model.save("s2s")

s = seq2seq(numEncoderTokens, numDecoderTokens, 64, "lstm", 3, 256, "lstm", 5, 256)
s.buildModel(encoder_input_data, decoder_input_data, decoder_target_data)



