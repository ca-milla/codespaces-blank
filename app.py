from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import uvicorn
from model import train_from_text, is_trained, generate_greedy, generate_sample, beam_search, vocab_size, make_model_state
import logging

"""Simple FastAPI application wrapping a tiny n-gram language model.

This module exposes three main routes:
    - GET `/` : render the single-page frontend
    - POST `/train` : train the in-memory model from a URL or local path
    - POST `/generate` : generate text from the model with configurable options

The `/generate` endpoint accepts a JSON body with options that control the
ngram order and decoding strategy. See the generate() docstring for field
details.
"""


app = FastAPI()

# mount static folder so templates can reference css/js
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')
# allow templates to use url_for('static', filename=...)
# Starlette's StaticFiles route uses a path parameter named 'path', but many
# templates call url_for('static', filename=...). Provide a
# small wrapper that maps filename->path so templates work without changes.
def _template_url_for(name: str, **path_params):
    if name == 'static' and 'filename' in path_params:
    # map 'filename' -> Starlette StaticFiles 'path'
        path_params['path'] = path_params.pop('filename')
    return app.url_path_for(name, **path_params)

templates.env.globals['url_for'] = _template_url_for

# create a model state and attach it to the FastAPI app state so it's explicit
app.state.model = make_model_state()


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request, 'title': 'LM demo'})


@app.post('/train')
async def train(request: Request):
    """Train the in-memory model.

    Expected JSON body:
        { "url": "<http-url-or-local-path>" }

    Behavior:
        - if `url` starts with 'http' the server will fetch it via requests
        - otherwise it will attempt to open the path as a local file

    Response JSON on success:
        { "ok": True, "vocab_size": <int>, "unigrams": <int>, "bigrams": <int> }

    On error returns a 400 with `detail` describing the problem.
    """
    data = await request.json()
    url = data.get('url')
    if not url:
        raise HTTPException(status_code=400, detail='missing url')

    try:
        if str(url).startswith('http'):
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            text = resp.text
        else:
            with open(url, 'r', encoding='utf-8') as f:
                text = f.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    train_from_text(app.state.model, text)
    return JSONResponse({'ok': True, 'vocab_size': vocab_size(app.state.model), 'unigrams': sum(1 for _ in app.state.model['unigrams']), 'bigrams': sum(1 for _ in app.state.model['bigrams'])})


@app.post('/generate')
async def generate(request: Request):
    """Generate text from the trained model.

    Expected JSON body fields (all optional):
      - prompt: (string) optional prompt; whitespace-tokenized
      - length: (int) maximum number of tokens to generate (default 20)
      - ngram: 'bigram' or 'trigram' (default 'trigram')
      - decoding: 'greedy' | 'sample' | 'beam' (default 'greedy')
      - beam_size: (int) used only when decoding=='beam'
      - top_k: (int) sampling-only parameter (keep top-k candidates)
      - top_p: (float) sampling-only nucleus parameter in (0,1)
      - temperature: (float) sampling-only temperature > 0

    Validation:
      - top_k/top_p/temperature are only accepted when decoding == 'sample'

    Response JSON on success:
      { "ok": True, "output": "generated text..." }

    Errors return a 400 status with a descriptive `detail`.
    """
    data = await request.json()
    prompt = data.get('prompt', '')
    length = int(data.get('length', 20))
    # ngram selection: 'bigram' or 'trigram'
    ngram = data.get('ngram', 'trigram')
    prefer_trigram = True if ngram == 'trigram' else False
    decoding = data.get('decoding', 'greedy')
    beam_size = int(data.get('beam_size', 3))
    top_k = int(data.get('top_k', 0))
    top_p = float(data.get('top_p', 0.0))
    temperature = float(data.get('temperature', 1.0))

    # validate sampling-only params
    if decoding != 'sample' and (top_k != 0 or (top_p != 0.0) or (temperature != 1.0)):
        raise HTTPException(status_code=400, detail='top_k, top_p and temperature are only valid with decoding=sample')

    if not is_trained(app.state.model):
        raise HTTPException(status_code=400, detail='model not trained yet')

    # dispatch to the requested decoding strategy
    if decoding == 'sample':
        out = generate_sample(app.state.model, prompt, max_len=length, prefer_trigram=prefer_trigram, top_k=top_k, top_p=top_p, temperature=temperature)
    elif decoding == 'beam':
        beams = beam_search(app.state.model, prompt, beam_size=beam_size, steps=length, prefer_trigram=prefer_trigram)
        # return the highest-scoring sequence joined as string
        if beams:
            best_seq = beams[0][1]
            out = ' '.join(best_seq)
        else:
            out = ''
    else:
        out = generate_greedy(app.state.model, prompt, max_len=length, prefer_trigram=prefer_trigram)

    # return exactly what generator produced
    return JSONResponse({'ok': True, 'output': out})


# minimal API: /, /train, /generate


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=5000, reload=True)
