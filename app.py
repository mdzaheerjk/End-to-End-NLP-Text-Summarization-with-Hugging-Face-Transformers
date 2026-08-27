from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response, HTMLResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import traceback
from src.textSummarizer.pipeline.prediction_pipeline import PredictionPipeline

pipeline_instance = None

def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = PredictionPipeline()
    return pipeline_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Pre-warming model into GPU / memory for instant inference...")
    try:
        get_pipeline()
        print("Model is ready for instant predictions!")
    except Exception as e:
        print(f"Warning: Could not pre-warm model on startup: {e}")
    yield

app = FastAPI(title="Text Summarization System", lifespan=lifespan)

# Mount static folder for CSS and assets
os.makedirs("static/css", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error occurred! {e}")

@app.post("/predict")
async def predict_route(request: Request, text: str = None):
    try:
        input_text = text
        if not input_text:
            try:
                data = await request.json()
                if isinstance(data, dict):
                    input_text = data.get("text", "")
                elif isinstance(data, str):
                    input_text = data
            except Exception:
                pass
        
        if not input_text:
            raw_body = await request.body()
            if raw_body:
                input_text = raw_body.decode("utf-8")

        if not input_text or not input_text.strip():
            return Response("Error: No input text provided", status_code=400)

        obj = get_pipeline()
        summary = obj.predict(input_text)
        return {"summary": summary, "text": summary}
    except Exception as e:
        traceback.print_exc()
        return Response(f"Error: {e}", status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)