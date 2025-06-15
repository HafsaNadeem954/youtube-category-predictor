import joblib

# Update these paths to your actual files
model_path = r"C:\Users\DELL\Documents\BDA_Project\model\YouTubeCategoryModel\best_rf_model.pkl"
scaler_path = r"C:\Users\DELL\Documents\BDA_Project\model\YouTubeCategoryModel\scaler.pkl"
encoder_path = r"C:\Users\DELL\Documents\BDA_Project\model\YouTubeCategoryModel\label_encoder.pkl"

# Load each
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)

print("Model loaded:", model)
print("Scaler loaded:", scaler)
print("Label Encoder loaded:", label_encoder)
