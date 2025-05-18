import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score
import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import Dense
from keras.utils import to_categorical


# Load and preprocess the data with advanced features
def load_and_preprocess_data(filename):
    data = pd.read_csv(filename)

    # Use the specified features
    features = [
        "length", "num_chars", "num_digits", "num_upper",
        "num_lower", "num_special", "num_vowels", "num_syllables"
    ]
    X = data[features]
    y = data["strength_label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_categorical, label_encoder, scaler

# Build the neural network model
def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main function
def main():
    X_scaled, y_categorical, label_encoder, scaler = load_and_preprocess_data('passwords.csv')
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=5)

    model = build_model(input_dim=X_train.shape[1], output_dim=y_categorical.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, validation_split=0.1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    precision = precision_score(y_test_labels, y_pred_labels, average='macro')

    print(f"📊 Accuracy: {accuracy * 100:.2f}%")
    print(f"🎯 Precision: {precision * 100:.2f}%")

    # Save model and encoder
    model.save("key_strength_nn_model.keras")
    joblib.dump(label_encoder, "label_encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Modèle neuronal et encodeur sauvegardés avec succès.")

if __name__ == "__main__":
    main()
