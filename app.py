from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import uvicorn
from model import train_from_text, is_trained, generate_greedy, vocab_size, make_model_state

app = FastAPI()

# mount static folder so templates can reference css/js
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

# create a model state and attach it to the FastAPI app state so it's explicit
app.state.model = make_model_state()


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request, 'title': 'LM demo'})


@app.post('/train')
async def train(request: Request):
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
    data = await request.json()
    prompt = data.get('prompt', '')
    length = int(data.get('length', 20))

    if not is_trained(app.state.model):
        raise HTTPException(status_code=400, detail='model not trained yet')

    out = generate_greedy(app.state.model, prompt, max_len=length)
    return JSONResponse({'ok': True, 'output': out})


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=5000, reload=True)
