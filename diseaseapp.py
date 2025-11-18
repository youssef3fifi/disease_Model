import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# 1. تهيئة التطبيق
app = Flask(__name__)

# 2. تحميل المودل والـ Encoders
print("Loading model and encoders...")
try:
    model = joblib.load("disease_model.pkl")
    encoders = joblib.load("disease_encoders.pkl")
    print("Model and Encoders loaded successfully.")
except Exception as e:
    print(f"Error loading files: {e}")
    model = None
    encoders = None

# دي الأعمدة اللي المودل اتدرب عليها (من الـ Notebook)
feature_cols = ["Chromosome", "Gene", "Variant_Type", "CLNSIG", "Risk_Prob", "Risk_Level"]
categorical_cols = ["Chromosome", "Gene", "Variant_Type", "CLNSIG", "Risk_Level"]

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or encoders is None:
        return jsonify({"error": "Model or encoders could not be loaded"}), 500

    try:
        # 1. هات الداتا الـ JSON
        input_data = request.get_json()
        
        # 2. حوّلها لـ DataFrame عشان الـ encoding
        df_input = pd.DataFrame([input_data])
        
        # 3. اعمل Encoding للأعمدة الكلام
        for col in categorical_cols:
            val = df_input[col].iloc[0]
            le = encoders[col]
            if val not in le.classes_:
                return jsonify({"error": f"Unseen value '{val}' for column '{col}'"}), 400
            
            # تحويل القيمة للرقم المقابل لها
            df_input[col] = le.transform(df_input[col])

        # 4. اتأكد إن الأعمدة مترتبة صح
        df_input = df_input[feature_cols]

        # 5. التنبؤ بالمرض (الرقم)
        pred_label = model.predict(df_input)[0]
        
        # 6. فك الـ Encoding (حوّل الرقم لاسم المرض)
        pred_disease = encoders["Disease"].inverse_transform([pred_label])[0]

        # 7. رجّع النتيجة
        return jsonify({
            "predicted_disease": pred_disease
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# (اختياري) عشان نتأكد إن السيرفر شغال
@app.route('/', methods=['GET'])
def home():
    return "Disease Prediction API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)