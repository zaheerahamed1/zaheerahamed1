
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers

# Setup logging
logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    logging.info("Loading claims data...")
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, numeric_cols):
    logging.info("Preprocessing data...")
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[numeric_cols])
    return df_scaled, scaler

def build_autoencoder(input_dim):
    logging.info("Building autoencoder...")
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation='relu', activity_regularizer=regularizers.l1(1e-5))(input_layer)
    encoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def detect_anomalies(autoencoder, data, threshold_factor=1.5):
    logging.info("Detecting anomalies...")
    reconstructions = autoencoder.predict(data)
    mse = np.mean(np.power(data - reconstructions, 2), axis=1)
    threshold = np.mean(mse) + threshold_factor * np.std(mse)
    anomalies = mse > threshold
    return anomalies, mse, threshold

if __name__ == "__main__":
    # Example CSV expected: claim_id, provider_id, billed_amount, procedure_code_1, ..., num_days
    file_path = "claims_data.csv"
    df = load_data(file_path)

    # Select numeric columns for model training
    numeric_cols = ['billed_amount', 'num_days']  # add more features as needed
    data_scaled, scaler = preprocess_data(df, numeric_cols)

    # Split data
    X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

    # Build and train model
    autoencoder = build_autoencoder(input_dim=X_train.shape[1])
    autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

    # Detect anomalies
    anomalies, mse, threshold = detect_anomalies(autoencoder, data_scaled)
    df['anomaly'] = anomalies
    df['reconstruction_error'] = mse

    # Save results
    df.to_csv("claims_anomaly_results.csv", index=False)
    logging.info(f"Anomaly detection completed. Threshold: {threshold}")
    print("Saved: claims_anomaly_results.csv")
