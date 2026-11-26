# Running the Fusha layer inside Saleem

The Fusha layer turns foreign words and Saudi dialect words into Modern Standard Arabic. In Saleem's editor its suggestions appear as green cards («كلمة أعجمية», «كلمة عامية») after the Kass cards. It runs as its own container behind Saleem's gateway, which forwards `POST /fusha/predict` to it. The text never leaves Saleem's servers.

## 1. What the container needs

| Item | Where it comes from | Path in the container |
|---|---|---|
| Code, training data, Saleem glossary | this repository (branch `KassFushaFeature`) | built into the image |
| Foreign word model (`foreign_to_msa_model` + the two CSV files beside it, 1.9 GB) | the shared Google Drive folder | mount at `/models/foreign_to_msa` |
| Word tagger (`tagger_merged`, 650 MB) | the shared Google Drive folder | mount at `/models/tagger_merged` |
| Gemma (`gemma-4-12B-it`) | Saleem's language model server | `FUSHA_LLM_URL` |

The two model folders are not in the image, so a new model needs no rebuild.

`/models/foreign_to_msa` must hold `foreign_to_msa_model/` (with `model.safetensors`), `trained_data_foriegn_to_msa.csv` and `val_data_foriegn_to_msa.csv`.

`/models/tagger_merged` must hold `model.safetensors`, `config.json` and the tokenizer files.

## 2. Settings

| Variable | Default in the image | Meaning |
|---|---|---|
| `FUSHA_LLM_API` | `openai` | `openai`: an OpenAI-compatible server (vLLM, TGI, Ollama `/v1`); `ollama`: Ollama's own API |
| `FUSHA_LLM_URL` | `http://gemma:8000` | base address of the Gemma server, without `/v1` |
| `FUSHA_LLM_MODEL` | `gemma-4-12B-it` | the model name that server expects |
| `FUSHA_LLM_KEY` | empty | bearer token, only if the Gemma server needs one |
| `FUSHA_TAGGER_DIR` | `/models/tagger_merged` | the word tagger |
| `FOREIGN_DIR` | `/models/foreign_to_msa` | the foreign word model and its data |

Gemma's thinking is switched off in every request (`chat_template_kwargs.enable_thinking=false`); otherwise the thinking uses up the answer's tokens and the layer gets an empty answer.

## 3. Build and run

```text
docker build -t fusha-layer:kass .
docker run -d --name ksgaal-fusha -p 8010:8010 \
  -v /srv/models/foreign_to_msa:/models/foreign_to_msa:ro \
  -v /srv/models/tagger_merged:/models/tagger_merged:ro \
  -e FUSHA_LLM_URL=http://<gemma host>:<port> \
  -e FUSHA_LLM_MODEL=gemma-4-12B-it \
  fusha-layer:kass
```

Model loading takes about a minute. Then `GET http://localhost:8010/health` answers `{"status": "healthy", ...}`.

Check one sentence:

```text
curl -X POST http://localhost:8010/predict -H "Content-Type: application/json" \
  -d '{"text": "خلاص بنرسل لك اللينك على الواتس"}'
```

The answer lists every replaced word with `type` (`foreign` or `dialect`), `msa`, `tier` (`suggested` or `review`) and `span` (word positions).

The container name `ksgaal-fusha` and port `8010` are what the gateway expects (`ocelot.json`, `ocelot.Production.json`). In Development the gateway uses `127.0.0.1:8010`.

## 4. Resources

- CPU only: the two small models need about 3 GB of memory.
- Each sentence takes about 2 to 7 seconds, mostly waiting for Gemma. The editor sends one sentence per request.

## 5. Tested

On 2026-11-26, on Windows, with the packages of `deploy/requirements-serve.txt` in a new Python 3.11 environment and the two models from local folders:

- Ollama API: same answers as the original setup.
- OpenAI-compatible API (Ollama's `/v1` with `gemma4:12b`): answers come back with thinking off.

Not tested: the Docker build itself (no Docker on that machine) and Saleem's own Gemma server.
