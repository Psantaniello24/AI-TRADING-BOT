import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np

class StockPredictor:
    def __init__(self, input_shape):
        """
        Initialize the LSTM model for stock price prediction
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, features)
        """
        self.model = self._build_model(input_shape)
        
    def _build_model(self, input_shape):
        """
        Build and compile the LSTM model
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, features)
            
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer with return sequences
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        
        # Third LSTM layer
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the LSTM model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of training data to use for validation
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        history = self.model.fit(
            X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Args:
            X (np.array): Input features
            
        Returns:
            np.array: Predicted values
        """
        return self.model.predict(X)
    
    def save_model(self, path):
        """
        Save the model to disk
        
        Args:
            path (str): Path to save the model
        """
        self.model.save(path)
    
    @classmethod
    def load_model(cls, path):
        """
        Load a saved model from disk
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            StockPredictor: Instance with loaded model
        """
        model = tf.keras.models.load_model(path)
        instance = cls((None, 1))  # Create instance with dummy input shape
        instance.model = model
        return instance 