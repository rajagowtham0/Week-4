from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Clinical Similarity API is running"}