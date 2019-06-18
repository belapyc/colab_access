import keras
from keras.layers import Input, Dense, Dropout
from keras.layers.core import Dense

def AE_train(dataframe, hidden_layer_size, epochs_amount, batch_amount, show_progress):
    """
    Creating and training single-layer autoencoder.
    Using Keras API for tensorflow environment.

    :dataframe: which dataframe to use for training a single-layer autoencoder
    :hidden_layer_size: amount of units in the hidden layer
    :epochs_amount: amount of epochs
    :batch_amount: batch size (16, 32, 64, 128, 256)
    :show_progress: 1 - to show epochs, 0 - to ignore epochs

    :Example:
    >>> encoder, decoder = AE_train(df, 10, 100, 256, 1)

    Author: Nikita Vasilenko
    """
    input_layer_size = dataframe.shape[1]
    # Creating input layer of the neural network
    input_sample = Input(shape=(input_layer_size,))
    # Second (hidden) layer of the neural network with ReLU activation function
    encoded = Dense(hidden_layer_size, activation='relu')(input_sample)
    # Third (output) layer of the neural network with Logistic Sigmoid activation function
    decoded = Dense(input_layer_size, activation='sigmoid')(encoded)
    # Initialising a model, mapping from input layer to the output layer (input -> reconstruction)
    autoencoder = Model(input_sample, decoded)
    # Splitting the model into two parts. Encoder - first two layers of the neural network
    encoder = Model(input_sample, encoded)
    # Creating an additional tensor (layer), effectively representing the encoded input (middle layer)
    encoded_input = Input(shape=(hidden_layer_size,))
    # Reconstructing decoder layer
    decoder_layer = autoencoder.layers[-1](encoded_input)
    # Create the decoder model
    decoder = Model(encoded_input, decoder_layer)
    # Compiling autoencoder model
    autoencoder.compile(optimizer='Adadelta', loss='binary_crossentropy')
    print('... [training in process] ...')
    # Training autoencoder
    autoencoder.fit(dataframe, dataframe,
                epochs=epochs_amount,
                batch_size=batch_amount,
                shuffle=True,
                verbose = show_progress)
    # Computing the training error (RMSE)
    hidden_layer = encoder.predict(dataframe)
    reconstruction_layer = decoder.predict(hidden_layer)
    #print("Training RMSE: ", find_mean_rmse(dataframe.as_matrix(),reconstruction_layer))
    return encoder, decoder

def AE_predict(encoder, decoder, df):
    """
    Given a trained mode, fit the data to the model and get output from the hidden layer

    :encoder: trained Encoder model
    :decoder: trained Decoder model
    :df: data to fit to the model

    :Example:
    >>> features = AE_predict(encoder, decoder, df)

    Author: Nikita Vasilenko
    """
    hidden_layer = encoder.predict(df)
#     The reconstruction layer is cast out (we dont need it anymore)
    reconstruction_layer = decoder.predict(hidden_layer)
    return hidden_layer

def SAE_train(dataframe, hidden_layer_size, epochs_amount, batch_amount, depth, show_progress):
    """
    Train a series (stack) of single-layer autoencoders

    :dataframe: which dataframe to use for training a single-layer autoencoder
    :hidden_layer_size: amount of units in the hidden layer
    :epochs_amount: amount of epochs
    :batch_amount: batch size (16, 32, 64, 128, 256)
    :show_progress: 1 - to show epochs, 0 - to ignore epochs

    :Example:
    >>> encoder, decoder = SAE_train(df, 10, 100, 256, 4, 1)

    Author: Nikita Vasilenko
    """
    encoders = []
    decoders = []
    print('Training AutoEncoder #1')
    encoder, decoder = AE_train(dataframe, hidden_layer_size, epochs_amount, batch_amount, show_progress)
    hidden_layer = AE_predict(encoder, decoder, dataframe)
    encoders.append(encoder)
    decoders.append(decoder)

    for i in range(0, depth - 1):
        print('Training AutoEncoder #', (i + 2))
        encoder, decoder = AE_train(pd.DataFrame(hidden_layer), hidden_layer_size, epochs_amount, batch_amount, show_progress)
        hidden_layer = AE_predict(encoder, decoder, hidden_layer)
        encoders.append(encoder)
        decoders.append(decoder)
    return encoders, decoders

def SAE_predict(encoders, decoders, dataframe):
    """
    Fit data to a trained stacked autoencoder

    :encoders: a LIST of trained encoders
    :decoders: a LIST of trained decoders
    :dataframe: data to fit to the model

    :Example:
    >>> features = SAE_predict(encoders, decoders, df)

    Author: Nikita Vasilenko
    """
    hidden_layer = AE_predict(encoders[0], decoders[0], dataframe)

    for i in range(1, len(encoders)):
        hidden_layer = AE_predict(encoders[i], decoders[i], hidden_layer)

    return hidden_layer
