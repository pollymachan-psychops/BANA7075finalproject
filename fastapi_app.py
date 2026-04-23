from fastapi import FastAPI
from src.schemas import PatientRequest, BatchPatientRequest
from src.recommender import load_artifacts, recommend_programs, recommend_programs_batch

app = FastAPI(title="SparkSync API")

pipeline, program_df = load_artifacts()


@app.get("/")
def root():
    return {"message": "SparkSync API is running"}


@app.post("/predict")
def predict(patient: PatientRequest):
    recommendations = recommend_programs(
        patient_input=patient.model_dump(),
        pipeline=pipeline,
        program_df=program_df,
        top_n=3,
    )
    return {"recommendations": recommendations}


@app.post("/predict_batch")
def predict_batch(batch: BatchPatientRequest):
    results = recommend_programs_batch(
        records=[record.model_dump() for record in batch.records],
        pipeline=pipeline,
        program_df=program_df,
        top_n=3,
    )
    return {"results": results}
