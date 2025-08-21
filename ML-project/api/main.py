from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from src.inference import predict_image

app = FastAPI(title="Spinach Freshness API", version="1.0.0")
origins = [
    "http://localhost:3000",  # React dev server
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Spinach Freshness API is running"}

@app.post("/predict-image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read())).convert("RGB")
        prediction = predict_image(img)  # should return a dict like {'label': 'Fresh', 'score': 0.92}
        # Always return JSON with 'success' flag
        return {"success": True, "result": prediction}
    except Exception as e:
        # Catch errors and return them to frontend
        return {"success": False, "error": str(e)}
