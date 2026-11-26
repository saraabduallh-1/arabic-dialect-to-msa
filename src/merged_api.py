"""HTTP service for the merged layer (for Saleem). Same shape as the foreign project's app.py.

The small models (tagger ~650 MB, foreign model ~1.9 GB) run on the CPU; Gemma is reached through Ollama, so
the layer can share the gemma4:12b that Saleem already runs instead of loading a second copy.

run:   uvicorn merged_api:app --app-dir src --host 0.0.0.0 --port 8010
call:  POST /predict {"text": "ابي اموديفاي العقد بكرة"}
       -> {"input", "msa", "replacements": [{"word", "type": dialect|foreign, "msa", "tier": suggested|review,
                                              "source", "span"}], "not_applied": [...]}
"""
from fastapi import FastAPI
from pydantic import BaseModel

from merged import Layer

app = FastAPI(title="Dialect + Foreign to MSA layer")
layer = Layer(device="cpu")


class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "healthy", "gemma": layer.gemma, "variant": layer.variant}


@app.post("/predict")
def predict(req: PredictRequest):
    o = layer.run([req.text])[0]
    keep = ("span", "word", "type", "msa", "tier", "source", "glossary")
    return {"input": req.text, "msa": o["msa"], "replacements": [{k: r[k] for k in keep} for r in o["replacements"]],
            "not_applied": o["not_applied"]}
