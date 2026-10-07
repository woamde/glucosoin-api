import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from google import genai
from openai import OpenAI

# Support MongoDB via PyMongo
try:
    from pymongo import MongoClient
    from bson import ObjectId
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GlycoSoinAPI")

# Chargement du .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="GlycoSoin API")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONNEXION MONGODB ---
DATABASE_URL = os.environ.get("DATABASE_URL")
mongo_client = None
db = None

if HAS_PYMONGO and DATABASE_URL:
    try:
        mongo_client = MongoClient(DATABASE_URL, serverSelectionTimeoutMS=5000)
        # On force explicitement le nom de la base de données Atlas
        db = mongo_client["glycosoin"]
        logger.info(f"Connexion MongoDB Atlas établie sur la base : {db.name}")
    except Exception as err:
        logger.error(f"Échec de connexion MongoDB : {err}")
else:
    logger.warning("PyMongo non disponible ou DATABASE_URL manquante. Mode mémoire actif.")
    except Exception as err:
        logger.error(f"Échec de connexion MongoDB : {err}")
else:
    logger.warning("PyMongo non disponible ou DATABASE_URL manquante. Mode mémoire actif.")

# --- BASE DE DONNÉES DE SECOURS (EN MÉMOIRE) ---
profile_db: Dict[str, Any] = {
    "first_name": "Abderahim",
    "name": "Abderahim",
    "diabetes_type": "DT1",
    "target_min": 70,
    "target_max": 180,
    "insulin_ratio": 10,
    "notes": "Suivi personnalisé"
}

journal_db_fallback: List[Dict[str, Any]] = [
    {
        "id": "1",
        "value": 118,
        "glucose": {"value": 118, "trend_arrow": "STABLE"},
        "timestamp": "2026-10-06T20:30:00Z",
        "insulin_units": 4,
        "meal_carbs": 45,
        "notes": "Dîner"
    },
    {
        "id": "2",
        "value": 142,
        "glucose": {"value": 142, "trend_arrow": "STABLE"},
        "timestamp": "2026-10-06T13:00:00Z",
        "insulin_units": 6,
        "meal_carbs": 60,
        "notes": "Déjeuner"
    }
]

def format_mongo_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    doc = dict(doc)
    if "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc

# --- MODÈLES PYDANTIC ---
class ChatRequest(BaseModel):
    model: str
    message: str

class ProfileModel(BaseModel):
    first_name: Optional[str] = "Abderahim"
    name: Optional[str] = "Abderahim"
    diabetes_type: Optional[str] = "DT1"
    target_min: Optional[float] = 70
    target_max: Optional[float] = 180
    insulin_ratio: Optional[float] = 10
    notes: Optional[str] = ""

class JournalEntryModel(BaseModel):
    id: Optional[str] = None
    value: Optional[float] = None
    glucose: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    insulin_units: Optional[float] = 0
    meal_carbs: Optional[float] = 0
    notes: Optional[str] = ""

# --- ROUTES API ---

@app.get("/")
async def root():
    return {"status": "ok", "message": "API GlycoSoin opérationnelle"}

# 1. Endpoints Profil
@app.get("/api/profile")
@app.get("/profile")
async def get_profile():
    if db is not None:
        try:
            profile_collection = db["profile"]
            data = profile_collection.find_one({})
            if data:
                return format_mongo_doc(data)
        except Exception as e:
            logger.error(f"Erreur de lecture profil MongoDB : {e}")
    return profile_db

@app.post("/api/profile")
@app.post("/profile")
async def save_profile(profile: ProfileModel):
    data = profile.dict(exclude_unset=True)
    if db is not None:
        try:
            profile_collection = db["profile"]
            profile_collection.replace_one({}, data, upsert=True)
            return data
        except Exception as e:
            logger.error(f"Erreur de sauvegarde profil MongoDB : {e}")
    profile_db.update(data)
    return profile_db

# 2. Endpoints Journal Glycémique
@app.get("/api/journal")
@app.get("/journal")
async def get_journal(limit: Optional[int] = None):
    if db is not None:
        try:
            collection = db["glucose_readings"]
            cursor = collection.find().sort("measured_at", -1)
            
            if limit and limit > 0:
                cursor = cursor.limit(limit)
                
            results = []
            for doc in cursor:
                # On mappe les clés réelles de MongoDB vers le format du frontend
                results.append({
                    "id": str(doc["_id"]),
                    "value": doc.get("value_mgdl"), 
                    "glucose": {
                        "value": doc.get("value_mgdl"),
                        "trend_arrow": doc.get("trend_arrow")
                    },
                    "timestamp": doc.get("measured_at"),
                    "source": doc.get("source"),
                    "insulin_units": doc.get("insulin_units", 0),
                    "meal_carbs": doc.get("meal_carbs", 0),
                    "notes": doc.get("notes", "")
                })
            
            if results:
                return results
        except Exception as e:
            logger.error(f"Erreur de lecture du journal MongoDB : {e}")

    if limit and limit > 0:
        return journal_db_fallback[:limit]
    return journal_db_fallback

@app.post("/api/journal")
@app.post("/journal")
async def add_journal_entry(entry: JournalEntryModel):
    data = entry.dict()
    if not data.get("glucose") and data.get("value"):
        data["glucose"] = {"value": data["value"], "trend_arrow": "STABLE"}

    if db is not None:
        try:
            collection = db["glucose_readings"]
            
            # Formatage pour correspondre à MongoDB (en cas d'ajout manuel depuis le frontend)
            mongo_data = {
                "value_mgdl": data.get("value"),
                "measured_at": data.get("timestamp"),
                "trend_arrow": data.get("glucose", {}).get("trend_arrow", ""),
                "insulin_units": data.get("insulin_units"),
                "meal_carbs": data.get("meal_carbs"),
                "notes": data.get("notes")
            }
            
            result = collection.insert_one(mongo_data)
            data["id"] = str(result.inserted_id)
            return data
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout MongoDB : {e}")

    data["id"] = str(len(journal_db_fallback) + 1)
    journal_db_fallback.insert(0, data)
    return data

@app.delete("/api/journal/{entry_id}")
@app.delete("/journal/{entry_id}")
async def delete_journal_entry(entry_id: str):
    if db is not None:
        try:
            collection = db["glucose_readings"]
            try:
                query = {"_id": ObjectId(entry_id)}
            except Exception:
                query = {"id": entry_id}
            
            result = collection.delete_one(query)
            if result.deleted_count > 0:
                return {"status": "success", "deleted_id": entry_id}
        except Exception as e:
            logger.error(f"Erreur lors de la suppression MongoDB : {e}")

    global journal_db_fallback
    journal_db_fallback = [item for item in journal_db_fallback if item.get("id") != entry_id]
    return {"status": "success", "deleted_id": entry_id}

# 3. Endpoint Assistant IA (Gemini / Claude / GPT)
@app.post("/api/ai/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        reply_content = ""
        requested_model = request.model.lower().strip()

        if "gemini" in requested_model:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise HTTPException(
                    status_code=400, 
                    detail="Clé GEMINI_API_KEY ou GOOGLE_API_KEY manquante."
                )
            
            client = genai.Client(api_key=api_key)
            models_to_try = [
                "gemini-3.5-flash-lite", 
                "gemini-3.5-flash", 
                "gemini-2.5-flash"
            ]
            
            if requested_model in models_to_try:
                models_to_try.remove(requested_model)
                models_to_try.insert(0, requested_model)

            last_error = None
            for model_id in models_to_try:
                try:
                    logger.info(f"Appel Gemini avec le modèle : {model_id}")
                    response = client.models.generate_content(
                        model=model_id,
                        contents=request.message,
                    )
                    reply_content = response.text
                    logger.info(f"Réponse Gemini générée avec succès via {model_id}")
                    break
                except Exception as err:
                    last_error = err
                    err_str = str(err)
                    logger.warning(f"Échec avec {model_id} : {err_str}")
                    fallback_triggers = ["503", "UNAVAILABLE", "404", "NOT_FOUND", "429", "RESOURCE_EXHAUSTED"]
                    if any(trigger in err_str for trigger in fallback_triggers):
                        continue
                    raise err
            else:
                raise HTTPException(
                    status_code=503, 
                    detail=f"Les serveurs Gemini sont indisponibles. ({str(last_error)})"
                )

        elif "claude" in requested_model:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="Clé ANTHROPIC_API_KEY manquante.")
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": request.message}],
            )
            reply_content = response.content[0].text

        elif "gpt" in requested_model:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="Clé OPENAI_API_KEY manquante.")
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": request.message}],
            )
            reply_content = response.choices[0].message.content

        else:
            raise HTTPException(status_code=400, detail=f"Modèle non reconnu : {request.model}")

        return {"reply": reply_content}

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Erreur serveur interne : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)