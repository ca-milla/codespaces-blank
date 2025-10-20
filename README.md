LM-demo — repo minimo per un piccolo language model (bigram)
===========================================================

Breve panoramica
----------------
Questo repository contiene una piccola applicazione web che permette di:
- addestrare un modello bigram da un file di testo,
- generare testo in modo greedy (bigram) a partire da un prompt.

L'app è volutamente minimale: la UI è una singola pagina che usa Vue/Vuetify via CDN per invocare le API.

Struttura dei file
------------------
- `app.py` — server FastAPI (rotte minime: `/`, `/train`, `/generate`). Monta la cartella `static` e rende template Jinja2.
- `model.py` — implementazione minimale del modello di bigrammi:
	- `make_model_state()` -> dict con `unigrams`, `bigrams`, `trained`;
	- `train_from_text(state, text)` -> aggiorna i conteggi in `state`;
	- `generate_greedy(state, start_token, max_len=20)` -> genera testo usando probabilità bigram con smoothing di Laplace.
- `templates/index.html` — frontend (single-page): due controlli per `train` e `generate` e area `Output` che mostra il risultato.
- `static/main.css` — stile minimo.
- `requirements.txt` — dipendenze Python.

Dipendenze e avvio
------------------
Questo progetto richiede Python 3.10+ (o 3.11/3.12). Le dipendenze principali sono elencate in `requirements.txt`.

Esempio di setup locale (Linux/macOS):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

L'app si avvia su `http://0.0.0.0:5000` (port configurato in `app.py`). In sviluppo `uvicorn` è avviato con `reload=True`.

Run (alternative e troubleshooting)
---------------------------------

Avviare con uvicorn (consigliato):

```bash
# usando il file app.py
uvicorn app:app --reload --host 0.0.0.0 --port 5000
```

Cambiare porta:

```bash
# es. usare la porta 8000
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Se vedi "Address already in use" significa che un altro processo sta già usando la porta. Per individuare e fermare il processo su Linux:

```bash
# mostra processi in ascolto sulla porta 5000
ss -ltnp | grep ':5000'
# oppure
lsof -i :5000
# termina il processo (sostituisci PID con il numero mostrato)
kill <PID>
```

Se non vuoi terminare il processo, avvia uvicorn su una porta diversa come mostrato sopra.

API (per sviluppatori backend)
------------------------------
1) GET `/` — pagina HTML (frontend). Non serve autenticazione.

2) POST `/train` — addestra il modello in memoria.
	 - Body JSON: `{ "url": "<http-or-local-path-or-data-url>" }`
		 - se `url` inizia con `http` viene usata `requests` per scaricare; altrimenti si apre il file locale.
		 - è anche supportato un `data:` inline per test veloci (es. `data:text/plain,Hello world`).
	 - Response JSON di successo:
		 ```json
		 {
			 "ok": true,
			 "vocab_size": 42,
			 "unigrams": 42,
			 "bigrams": 123
		 }
		 ```
	 - Errori: 400 con `detail` se manca `url` o la fetch/file fallisce.

3) POST `/generate` — genera testo dal modello addestrato.
	 - Body JSON: `{ "prompt": "optional prompt", "length": 20 }`
	 - Response JSON:
		 ```json
		 { "ok": true, "output": "testo generato..." }
		 ```
	 - Errori: 400 se il modello non è addestrato (`model not trained yet`).

Formati e contratti (breve)
---------------------------
- Stato modello (`app.state.model`) è un dict:
	- `unigrams`: dict mapping token -> count (int)
	- `bigrams`: dict mapping tuple(token_prev, token_next) -> count (int)
	- `trained`: bool

- `generate_greedy` restituisce sempre una stringa (può essere vuota se non ci sono continuazioni osservate).

Comportamento del generatore
----------------------------
- Greedy bigram con smoothing di Laplace:
	P(w | prev) = (count(prev,w) + 1) / (count(prev) + V)
- Se non esistono bigram con `prev`, la generazione si ferma.
- I token speciali `<s>` e `</s>` vengono usati internamente per segnare inizio/fine frase.

Esempi di richieste (curl)
--------------------------
- Train con testo inline:

```bash
curl -s -X POST -H "Content-Type: application/json" \
	-d '{"url":"data:text/plain,Hello world. Hello again!"}' \
	http://localhost:5000/train | jq .
```

- Generate (dopo train):

```bash
curl -s -X POST -H "Content-Type: application/json" \
	-d '{"prompt":"Hello","length":20}' \
	http://localhost:5000/generate | jq .
```

Nota: se vuoi simulare la chiamata senza il frontend, usa curl come sopra.

Frontend (note rapide)
--------------------------------------------
- La UI è minima e usa Vue + Vuetify via CDN; la logica client è in `templates/index.html`.
- Il client invia JSON a `/train` e `/generate` e mostra il campo `output` restituito dal server.
- Se l'output non appare nell'interfaccia, apri DevTools → Network e Console per vedere la richiesta e la risposta, oppure chiama direttamente `/generate` con curl per verificare il JSON.


