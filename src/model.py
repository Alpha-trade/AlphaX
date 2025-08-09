from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape):
    """
    Creates, compiles, and returns an LSTM model for stock prediction.

    Args:
        input_shape (tuple): The shape of the input data (timesteps, n_features).

    Returns:
        A compiled Keras model.
    """
    model = Sequential()

    # First LSTM layer with Dropout
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Second LSTM layer with Dropout
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Dense layer
    model.add(Dense(units=25))

    # Output layer
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # Example of how to create the model
    # This is for demonstration purposes only.
    example_input_shape = (60, 5) # (timesteps, features)
    model = create_lstm_model(example_input_shape)
    model.summary()
