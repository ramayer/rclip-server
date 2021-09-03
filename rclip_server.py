#!/usr/bin/env python
import argparse
import clip
import clip.model
import functools
import html
import io
import json
import numpy as np
import pathlib
import PIL as pillow
import requests.utils
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
        rclip_sql =  f'SELECT filepath, vector, id FROM images WHERE filepath LIKE ? AND deleted IS NULL'
        return con.execute(rclip_sql,(path + '/%',))

def censor(path):
    with sqlite3.connect(args.db) as con:
        con.row_factory = sqlite3.Row
        rclip_sql =  f'UPDATE images SET deleted=True where filepath = ?'
        return con.execute(rclip_sql,(path,))

def _get_features(directory: str) -> Tuple[List[str], np.ndarray]:
    filepaths: List[str] = []
    features: List[np.ndarray] = []
    img_ids: List[int] = []
    for image in load_vecs(directory):
      filepaths.append(image['filepath'])
      features.append(np.frombuffer(image['vector'], np.float32))
      img_ids.append(image['id'])
    if not filepaths:
      return [], np.ndarray(shape=(0, Model.VECTOR_SIZE))
    return filepaths, np.stack(features), img_ids

################################################################################
# Web server
################################################################################

parser                    = argparse.ArgumentParser()
def_db                    = pathlib.Path.home() / '.local/share/rclip/db.sqlite3'
parser.add_argument('--db','-d',default=str(def_db),help="path to rclip's database")
args,unk                  = parser.parse_known_args()

filepaths:        List[str]        = []
global_features:  List[np.ndarray] = []

filepaths,global_features,img_ids   = _get_features('')
feature_minimums:np.ndarray = functools.reduce(lambda x,y: np.minimum(x,y), global_features)
feature_maximums:np.ndarray = functools.reduce(lambda x,y: np.maximum(x,y), global_features)
feature_ranges:np.ndarray   = feature_maximums - feature_minimums
idx_to_imgid = dict(enumerate(img_ids))
imgid_to_idx = dict([(y,x) for x,y in enumerate(img_ids)])
def get_path_from_id(img_id:int):
    return filepaths[imgid_to_idx[img_id]]
app = FastAPI()

###############################################################################
# HTTP Endpoints
###############################################################################
@app.get("/conceptmap", response_class=HTMLResponse)
async def search(q:str, m:str=None, p:str=None,num:int = 36, size:int=400, debug:bool=False):
    features = model.compute_text_features(q)
    if p:
      plus_features = model.compute_text_features(p)
      features = features + plus_features
    if m:
      minus_features = model.compute_text_features(m)
      features = features - minus_features
    combined_features = features
    print(features)
    results = compute_similarities(global_features, combined_features)
    print(results[0:10])
    return make_html(results,q,size,num,debug_features = combined_features)

@app.get("/search", response_class=HTMLResponse)
async def search(q:str, num:int = 36, size:int=400, debug:bool=False):
    "Accept either a json array of CLIP embedding values, or a search string"
    if q.startswith('['):
        text_features = np.asarray([json.loads(q)])
    else:
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
async def mlt(img_id:int,num:int=100,size=400):
    img_path = get_path_from_id(img_id)
    img = Image.open(img_path)
    ref_features = model.compute_image_features([img])[0]
    results = compute_similarities(global_features, [ref_features])
    return make_html(results,img_path,size,num,debug_features = ref_features)

@app.get("/mut", response_class=HTMLResponse)
async def mut(q:str,num:int=100,size=400):
    img = Image.open(q)
    ref_features = model.compute_image_features([img])[0]
    results = compute_similarities(global_features, [ref_features])
    results.reverse()
    return make_html(results,q,size,num,debug_features = ref_features)

@app.get("/censor", response_class=HTMLResponse)
async def censor_endpoint(img_id:int):
    path = get_path_from_id(img_id)
    censor(path)
    return(f"<html>Ok, {path} is censored</html>")

@app.get("/rnd", response_class=HTMLResponse)
async def rnd(seed=0,num:int=100,size:int=400):
    ref_features = np.random.rand(512) * feature_ranges + feature_minimums
    ref_features /= np.linalg.norm(ref_features)
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
async def img(img_id:int):
    img_path = get_path_from_id(img_id)
    hdrs = {
      'Cache-Control': 'public, max-age=172800'
    }
    return FileResponse(img_path, headers=hdrs)

@app.get("/thm")
async def thm(img_id:int, size:int=400):
    img_path = get_path_from_id(img_id)
    img = Image.open(img_path)
    thm = img.thumbnail((size,3*size/4))
    buf = io.BytesIO()
    img.save(buf,format="jpeg")
    buf.seek(0)
    hdrs = {
        'Cache-Control': 'public, max-age=172800'
    }
    return StreamingResponse(buf,media_type="image/jpeg", headers=hdrs)

###############################################################################
# Minimal HTML template
###############################################################################

def dedup_sims(similarities):
    seen = set()
    return [seen.add(s[0]) or s for s in similarities if s[0] not in seen and not seen.add(s[0])]

def make_html(similarities,q,size,num,debug_features=None,debug=False):
    sims = dedup_sims(similarities[:num*2])
    scores_with_imgids = [(score,idx_to_imgid[idx]) for score,idx in sims[:num]]
    imgs = [f"""
             <div style="">
                <a href="/img?img_id={img_id}" target="_blank"><img src="/thm?img_id={img_id}&size={size}"></a>
                <br>
                {str(int(100*s))+'% similarity'}
                <a href="/mlt?img_id={img_id}">more like this</a>
                <!--<a href="/censor?path=FIXME">censor this</a>-->
                {debug and "" or "<!--"} | <a href="/mut?q={img_id}">unlike this</a> {debug and "" or "-->"}
             </div>
             """ 
            for s,img_id in scores_with_imgids
            ]
    tmpl = string.Template("""<html">
       <style>
          body {background-color: #ccc; width: 100%; font-family:Ariel}
          form {margin:0px}
          #header,#footer {background-color: #888; padding: 30px; }
          .images div{display:inline-block; width:${__size__}px}
          #q {width:800px}
          #lq {font-size: 20pt}
          #sizes,.images {font-size: 10pt}
          a:link {text-decoration: none}
          a:hover {text-decoration: underline}
       </style>
       <div id="header">
       <form action="search">
           <table><tr>
            <td><label for="q" id="lq"><a href="/" style="color:black; text-decoration: none">Search:</a></label></td>
            <td><input name="q" id='q' value="$__q__" style="width: width:800px"></td>
            <td>
             <!--
             <input type="hidden" name="num"  id='num'   value="$__num__" style="width:30px">
             <input type="hidden" name="size" id='size'  value="$__size__" style="width:30px">
             -->
             <input type="submit" value="Go">
            </td>
           </tr><tr><td></td>
            <td id="sizes">
             <a href="$__tiny__">tiny</a>
             <a href="$__small__">small</a>
             <a href="$__medium__">medium</a>
             <a href="$__large__">large</a>
             <a href="$__huge__">huge</a>
            </td>
           </tr></table>
       </form>
       </div>
       <div class="images">
       $__imgs__
       </div>
       <script>
          document.getElementById("q").focus();
       </script>
       <br>
       <div id="footer">
       <a href="$__more__">more</a> | 
       <a href="$__opposite__">opposite</a>
       <br>
       </div>
       $__debug_txt__
    </html>""")
    debug_txt = ""
    if debug_features is not None:
        clip_vec_as_json = json.dumps(debug_features.flatten().tolist())
        debug_txt += f"<a href=search_by_json?q={urllib.parse.quote(clip_vec_as_json)}>CLIP embedding</a>:"
        debug_txt += "<table><tr>"
        normalized_debug_features = 255 * (debug_features - feature_minimums) / feature_ranges
        zipped_features = zip(debug_features.flatten(),normalized_debug_features.flatten())
        for idx,(df,nf) in enumerate(zipped_features):
            clr = nf > 255 and 255 or int(nf) < 0 and 0 or int(nf)
            debug_txt += f"""<td style="background:#{clr:02x}{clr:02x}ff">{float(df):0.2g}</td>"""
            if idx % 16 == 15: debug_txt += "</tr><tr>" 
        debug_txt += "</table>"
    bigger_num = (num > 100) and 1000 or num * 10
    return tmpl.substitute(__imgs__      = " ".join(imgs),
                           __q__         = html.escape(q),
                           __opposite__  = f"opposite?q={urllib.parse.quote(q)}&num={num}&size={size}",
                           __more__      = f"search?q={urllib.parse.quote(q)}&num={bigger_num}&size={size}",
                           __tiny__      = f"search?q={urllib.parse.quote(q)}&size=100&num=360",
                           __small__     = f"search?q={urllib.parse.quote(q)}&size=200&num=180",
                           __medium__    = f"search?q={urllib.parse.quote(q)}&size=400&num=48",
                           __large__     = f"search?q={urllib.parse.quote(q)}&size=600&num=24",
                           __huge__      = f"search?q={urllib.parse.quote(q)}&size=800&num=12",
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
