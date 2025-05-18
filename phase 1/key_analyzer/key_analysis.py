import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import re

class KeyAnalysis:
    model_path = "key_strength_nn_model.h5"
    encoder_path = "label_encoder.pkl"
    scaler_path = "scaler.pkl"

    try:
        model = load_model(model_path)
        label_encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
        print("✅ Modèle, encodeur et scaler chargés avec succès.")
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement : {e}")
        model = None
        label_encoder = None
        scaler = None

    @staticmethod
    def analyze_key(key: str) -> str:
        if KeyAnalysis.model is None or KeyAnalysis.label_encoder is None or KeyAnalysis.scaler is None:
            raise ValueError("❌ Le modèle, l'encodeur ou le scaler n'est pas chargé !")

        # Extraire les features avancées
        features = KeyAnalysis.extract_features(key)
        df = pd.DataFrame([features], columns=[
            "length", "num_chars", "num_digits", "num_upper",
            "num_lower", "num_special", "num_vowels", "num_syllables"
        ])

        scaled_features = KeyAnalysis.scaler.transform(df)
        prediction = KeyAnalysis.model.predict(scaled_features)
        predicted_index = np.argmax(prediction)
        predicted_label = KeyAnalysis.label_encoder.inverse_transform([predicted_index])[0]

        return f"🔍 Analyse de la clé : {predicted_label}"

    @staticmethod
    def extract_features(key: str):
        vowels = "aeiouAEIOU"
        special_chars = r"[!@#$%^&*(),.?\":{}|<>]"

        length = len(key)
        num_chars = len(set(key))
        num_digits = sum(c.isdigit() for c in key)
        num_upper = sum(c.isupper() for c in key)
        num_lower = sum(c.islower() for c in key)
        num_special = len(re.findall(special_chars, key))
        num_vowels = sum(c in vowels for c in key)
        num_syllables = KeyAnalysis.estimate_syllables(key)

        return [
            length, num_chars, num_digits, num_upper,
            num_lower, num_special, num_vowels, num_syllables
        ]

    @staticmethod
    def estimate_syllables(text: str) -> int:
        # Simple estimation : compte les groupes de voyelles
        return len(re.findall(r'[aeiouyAEIOUY]+', text))

if __name__ == "__main__":
    test_key = "1234@Zakaria//"
    verdict = KeyAnalysis.analyze_key(test_key)
    print(verdict)
