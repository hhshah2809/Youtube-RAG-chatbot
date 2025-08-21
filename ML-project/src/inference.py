import joblib
from PIL import Image
from .features import calculate_leaf_features

model = joblib.load("model.joblib")

def predict_image(image: Image.Image):
    feats = calculate_leaf_features(image)
    if feats is None:
        return {"ok": False, "reason": "no leaf detected"}
    X = [[feats["gcv"], feats["area"], feats["aspect_ratio"], feats["roundness"]]]
    pred = int(model.predict(X)[0])
    proba = float(max(model.predict_proba(X)[0]))
    return {"ok": True, "pred": pred, "fresh": bool(pred==0), "proba": proba, "features": feats}
