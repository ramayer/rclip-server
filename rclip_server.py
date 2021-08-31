#!/usr/bin/env python
import argparse
import pathlib
import clip
import clip.model
import html
import io
import numpy as np
import PIL as pillow
import requests.utils
import functools
import sqlite3
import string
import torch
import torch.nn
import urllib.parse
import uvicorn

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from PIL import Image
from starlette.responses import StreamingResponse
from tqdm import tqdm
from typing import Callable, List, Tuple, cast
from typing import Iterable, List, NamedTuple, Tuple, TypedDict, cast


###############################################################################
# Plagerized from rclip
###############################################################################

class Model:
  VECTOR_SIZE = 512
  _device = 'cpu'
  _model_name = 'ViT-B/32'

  def __init__(self):
    model, preprocess = cast(
      Tuple[clip.model.CLIP, Callable[[Image.Image], torch.Tensor]],
      clip.load(self._model_name, device=self._device)
    )
    self._model = model
    self._preprocess = preprocess

  def compute_image_features(self, images: List[Image.Image]) -> np.ndarray:
    images_preprocessed = torch.stack([self._preprocess(thumb) for thumb in images]).to(self._device)
    with torch.no_grad():
      image_features = self._model.encode_image(images_preprocessed)
      image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

  def compute_text_features(self, text: str) -> np.ndarray:
    with torch.no_grad():
      text_encoded = self._model.encode_text(clip.tokenize(text).to(self._device))
      text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    return text_encoded.cpu().numpy()

###############################################################################
# Paraphrased from rclip
###############################################################################

model = Model()

def compute_similarities(item_features: np.ndarray, desired_features: np.ndarray) -> List[Tuple[float, int]]:
    similarities = (desired_features @ item_features.T).squeeze(0).tolist()
    sorted_similarities = sorted(zip(similarities, range(item_features.shape[0])), key=lambda x: x[0], reverse=True)
    return sorted_similarities

def load_vecs(path):
    with sqlite3.connect(args.db) as con:
        con.row_factory = sqlite3.Row
        rclip_sql =  f'SELECT filepath, vector FROM images WHERE filepath LIKE ? AND deleted IS NULL'
        return con.execute(rclip_sql,(path + '/%',))

def censor(path):
    with sqlite3.connect(args.db) as con:
        con.row_factory = sqlite3.Row
        rclip_sql =  f'UPDATE images SET deleted=True where filepath = ?'
        return con.execute(rclip_sql,(path,))

def _get_features(directory: str) -> Tuple[List[str], np.ndarray]:
    filepaths: List[str] = []
    features: List[np.ndarray] = []
    for image in load_vecs(directory):
      filepaths.append(image['filepath'])
      features.append(np.frombuffer(image['vector'], np.float32))
    if not filepaths:
      return [], np.ndarray(shape=(0, Model.VECTOR_SIZE))
    return filepaths, np.stack(features)


################################################################################
# Web server
################################################################################

parser                    = argparse.ArgumentParser()
def_db                    = pathlib.Path.home() / '.local/share/rclip/db.sqlite3'
parser.add_argument('--db','-d',default=str(def_db),help="path to rclip's database")
args                      = parser.parse_args()

filepaths:        List[str]        = []
global_features:  List[np.ndarray] = []

filepaths,global_features   = _get_features('')
feature_minimums:np.ndarray = functools.reduce(lambda x,y: np.minimum(x,y), global_features)
feature_maximums:np.ndarray = functools.reduce(lambda x,y: np.maximum(x,y), global_features)
feature_ranges:np.ndarray   = feature_maximums - feature_minimums

app = FastAPI()

###############################################################################
# HTTP Endpoints
###############################################################################
@app.get("/search", response_class=HTMLResponse)
async def search(q:str, num:int = 30, size:int=400, debug:bool=False):
    text_features = model.compute_text_features(q)
    results = compute_similarities(global_features, text_features)
    return make_html(results,q,size,num,debug_features = text_features)

@app.get("/opposite", response_class=HTMLResponse)
async def opposite(q:str, num:int = 20, size:int=400):
    text_features = model.compute_text_features(q)
    results = compute_similarities(global_features, text_features)
    results.reverse()
    return make_html(results,q,size,num,debug_features = text_features)

@app.get("/mlt", response_class=HTMLResponse)
async def mlt(q:str,num:int=100,size=400):
    img = Image.open(q)
    ref_features = model.compute_image_features([img])[0]
    results = compute_similarities(global_features, [ref_features])
    return make_html(results,q,size,num,debug_features = ref_features)

@app.get("/censor", response_class=HTMLResponse)
async def censor_endpoint(path:str):
    censor(path)
    return(f"<html>Ok, {path} is censored</html>")

@app.get("/mut", response_class=HTMLResponse)
async def mut(q:str,num:int=100,size=400):
    img = Image.open(q)
    ref_features = model.compute_image_features([img])[0]
    results = compute_similarities(global_features, [ref_features])
    results.reverse()
    return make_html(results,q,size,num,debug_features = ref_features)

@app.get("/rnd", response_class=HTMLResponse)
async def rnd(seed=0,num:int=100,size:int=400):
    ref_features = np.random.rand(512) * feature_ranges + feature_minimums
    results = compute_similarities(global_features, [ref_features])
    return make_html(results,'[random]',size,num,debug_features = ref_features)

@app.get("/avg", response_class=HTMLResponse)
async def avg(seed=0,num:int=100):
    ref_features = 0.5 * feature_ranges + feature_minimums
    results = compute_similarities(global_features, [ref_features])
    return make_html(results,q,size,num,debug_features = ref_features)

@app.get("/",response_class=HTMLResponse)
def home():
    return make_html([],'',400,20,debug_features = None)

import os
@app.get("/img")
async def main(q:str):
    if os.access(q, os.R_OK):
        return FileResponse(q)
    else:
        censor(q)
        return HTMLResponse("Permission Denied")

@app.get("/thm")
async def main(q:str,size:int=400):
    if not os.access(q, os.R_OK):
        censor(q)
    img = Image.open(q)
    thm = img.thumbnail((size,size))
    buf = io.BytesIO()
    img.save(buf,format="jpeg")
    buf.seek(0)
    hdrs = {
        'Cache-Control': 'public, max-age=3600'
    }
    return StreamingResponse(buf,media_type="image/jpeg", headers=hdrs)


###############################################################################
# Minimal HTML template
###############################################################################

def make_html(similarities,q,size,num,debug_features=None,debug=False):
    scores_with_paths = [(score,filepaths[idx]) for score,idx in similarities[:num]]
    imgs = [f"""
             <div style="display:inline-block; border:1px solid black; width:{size}px">
                <a href="/img?q={urllib.parse.quote(i)}" target="_blank"><img src="/thm?q={urllib.parse.quote(i)}&size={size}"></a>
                <br>
                {debug and str(int(100*s))+'%' or ""}
                <a href="/mlt?q={urllib.parse.quote(i)}">more like this</a>
                <!--<a href="/censor?path={urllib.parse.quote(i)}">censor this</a>-->
                {debug and "" or "<!--"} | <a href="/mut?q={urllib.parse.quote(i)}">unlike this</a> {debug and "" or "-->"}
             </div>
             """ 
            for s,i in scores_with_paths
            ]
    tmpl = string.Template("""<html style="background-color:#444;">
       <div style="background-color: #666; width: 100%; padding: 30px; font-family:Ariel">
       <form action="search" style="font-size: 20pt">
           <label for="q"><a href="/" style="color:black; text-decoration: none">Search:</a></label>
           <input name="q"    id='q'     value="$__q__"    style="width:800px">
           <input type="hidden" name="num"  id='num'   value="$__num__"  style="width:30px">
           <input type="hidden" name="size" id='size'  value="$__size__" style="width:30px">
           <input type="submit" value="Go">
           <br>
       </form>
       <a href="$__tiny__">tiny</a>
       <a href="$__small__">small</a>
       <a href="$__medium__">medium</a>
       <a href="$__large__">large</a>
       <a href="$__huge__">huge</a>
       </div>
       $__imgs__
       <script>
          document.getElementById("q").focus();
       </script>
       <br>
       <div style="background-color: #ccc; width: 100%; padding: 30px; font-family:Ariel">
       <a href="$__more__">more</a> | 
       <a href="$__opposite__">opposite</a>
       <br>
       </div>
       $__debug_txt__
    </html>""")
    debug_txt = ""
    if debug_features is not None:
        normalized_debug_features = 100 * (debug_features - feature_minimums) / feature_ranges
        debug_txt = " ".join([f"{int(x):02d}" for x in normalized_debug_features.flatten()])
    bigger_num = (num > 100) and 1000 or num * 10
    return tmpl.substitute(__imgs__      = " ".join(imgs),
                           __q__         = html.escape(q),
                           __opposite__  = f"opposite?q={urllib.parse.quote(q)}&num={num}&size={size}",
                           __more__      = f"search?q={urllib.parse.quote(q)}&num={bigger_num}&size={size}",
                           __tiny__      = f"search?q={urllib.parse.quote(q)}&size=100&num=200",
                           __small__     = f"search?q={urllib.parse.quote(q)}&size=200&num=100",
                           __medium__    = f"search?q={urllib.parse.quote(q)}&size=400&num=20",
                           __large__     = f"search?q={urllib.parse.quote(q)}&size=600&num=10",
                           __huge__      = f"search?q={urllib.parse.quote(q)}&size=800&num=10",
                           __num__       = num,
                           __size__      = size,
                           __debug_txt__ = debug_txt
                           )

###############################################################################
# Launch the server
###############################################################################

if __name__ == "__main__":
    uvicorn.run("rclip_server:app", host='0.0.0.0', port=8000, debug=True, reload=True)


# ~/proj/rclip/bin/rclip.sh -n dog -f | feh -f - t -g 1000x1000
