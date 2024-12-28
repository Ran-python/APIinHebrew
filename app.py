from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import uuid
import pandas as pd
from gtts import gTTS
import pytesseract
import fitz  # PyMuPDF
from PIL import Image

app = FastAPI(
    title="OMERIKI Chatbot",
    description="A Hebrew Text-to-Speech, PDF-to-Speech, and Handwritten Recognition Tool.",
    version="1.0",
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Ensure dictionary file exists
CSV_FILE = 'omeriki_dictionary.csv'
if not os.path.exists(CSV_FILE):
    raise RuntimeError(f"File '{CSV_FILE}' does not exist. Place it in the project root.")

dict_df = pd.read_csv(CSV_FILE)

# Directory for generated audio files
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def index(request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dict/{word}", response_class=HTMLResponse)
async def dictionary_lookup(request, word: str):
    result = dict_df[dict_df['word'] == word]
    if result.empty:
        raise HTTPException(status_code=404, detail="Word not found in the dictionary.")
    
    definition = result.iloc[0]['definition']
    audio_file = result.iloc[0]['audio_file']

    return templates.TemplateResponse("dict.html", {
        "request": request,
        "word": word,
        "definition": definition,
        "audio_file": audio_file
    })

@app.post("/tts")
async def text_to_speech(text: str, slow: bool = Query(False)):
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    try:
        audio_file_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.mp3")
        tts = gTTS(text=text, lang='he', slow=slow)
        tts.save(audio_file_path)

        return {
            "message": "Audio file generated successfully!",
            "filename": os.path.basename(audio_file_path),
            "download_link": f"/static/{os.path.basename(audio_file_path)}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

@app.post("/pdf")
async def pdf_to_speech(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    try:
        pdf_document = fitz.open(stream=file.file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in pdf_document])
        pdf_document.close()

        return await text_to_speech(text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/handwritten")
async def handwritten_to_speech(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    try:
        image = Image.open(file.file)
        text = pytesseract.image_to_string(image, lang='heb')
        
        if not text.strip():
            raise HTTPException(status_code=404, detail="No text found in the uploaded image.")

        return await text_to_speech(text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
