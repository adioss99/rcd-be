import uvicorn
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from core.controller.main_controller import predict, delete, remove_folder
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(_:FastAPI):
    try:
        print("===Starting up===")
        scheduler = BackgroundScheduler()
        scheduler.add_job(remove_folder, 'cron', hour=23, minute=59)
        scheduler.start()
        yield
    finally:
        print("---Shutting down---")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def index():
    return JSONResponse({"message": "RDC API ready"})

@app.post("/predict")
async def upload(image: UploadFile = File(...)): 
    return await predict(image)

@app.delete("/delete/{img_id}")
async def delete_image(img_id: str):
    return delete(img_id)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST"),
        port=int(os.getenv("PORT")),
        reload=True
    )