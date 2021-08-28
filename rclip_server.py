#!/usr/bin/env python

import uvicorn
import html
import string
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse
import io
import requests.utils
import subprocess
import PIL as pillow
from PIL import Image
import urllib.parse

###############################################################################
# Plagerized from rclip
###############################################################################
from typing import Callable, List, Tuple, cast

import clip
import clip.model
import numpy as np
from PIL import Image
import torch
import torch.nn

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

  def compute_similarities_to_text(self, item_features: np.ndarray, text: str) -> List[Tuple[float, int]]:
    text_features = self.compute_text_features(text)

    similarities = (text_features @ item_features.T).squeeze(0).tolist()
    sorted_similarities = sorted(zip(similarities, range(item_features.shape[0])), key=lambda x: x[0], reverse=True)

    return sorted_similarities

###############################################################################
# Paraphrased from rclip
###############################################################################

model = Model()
# from rclip
import sqlite3
import numpy as np
from tqdm import tqdm
from typing import Iterable, List, NamedTuple, Tuple, TypedDict, cast

def load_vecs(path):
    con = sqlite3.connect("/home/ron/.local/share/rclip/db.sqlite3")
    con.row_factory = sqlite3.Row
    rclip_sql =  f'SELECT filepath, vector FROM images WHERE filepath LIKE ? AND deleted IS NULL'
    return con.execute(rclip_sql,(path + '/%',))

def _get_features(directory: str) -> Tuple[List[str], np.ndarray]:
    filepaths: List[str] = []
    features: List[np.ndarray] = []
    for image in load_vecs(directory):
      filepaths.append(image['filepath'])
      features.append(np.frombuffer(image['vector'], np.float32))
    if not filepaths:
      return [], np.ndarray(shape=(0, Model.VECTOR_SIZE))
    return filepaths, np.stack(features)

path = '/home/ron/tmp/old_pics_indexed_with_rclip/pics'
filepaths,features = _get_features(path)

def get_results(q,num):
    similarities = model.compute_similarities_to_text(features,q)
    paths = [filepaths[row[1]] for row in similarities[:num]]
    return paths


################################################################################
# Web server
################################################################################

app = FastAPI()

# https://stackoverflow.com/questions/55873174/how-do-i-return-an-image-in-fastapi
@app.get("/img")
async def main(q:str):
    path = q
    return FileResponse(path)

@app.get("/thm")
async def main(q:str,size:int=400):
    img = Image.open(q)
    thm = img.thumbnail((size,size))
    buf = io.BytesIO()
    img.save(buf,format="jpeg")
    buf.seek(0)
    hdrs = {
        'Cache-Control': 'public, max-age=3600'
    }

    return StreamingResponse(buf,media_type="image/jpeg", headers=hdrs)

@app.get("/search", response_class=HTMLResponse)
async def search(q:str, num:int = 20, size:int=400):
    if (slow_way := False):
        path = "/home/ron/tmp/old_pics_indexed_with_rclip/pics"
        cmd = f"(cd {path}; ~/proj/rclip/bin/rclip.sh -n --top 100 -f '{q}')"
        resp =  subprocess.run(cmd,shell=True,text=True,capture_output=True)
        result = resp.stdout
        results = result.splitlines()
    else:
        results = get_results(q,num)
    imgs = [f"""<a href="/img?q={urllib.parse.quote(i)}" target="_blank"><img src="/thm?q={urllib.parse.quote(i)}&size={size}"></a>""" 
            for i in results
            ]
    tmpl = string.Template("""<html>
       <div style="background-color: #ccc; width: 100%; padding: 30px; font-family:Ariel">
       <form action="search" style="font-size: 20pt">
           <label for="q">Search:</label>
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
       <a href="$__more__">more</a><br>
       </div>
    </html>""")
    bigger_num = (num > 100) and 1000 or num * 10
    return tmpl.substitute(__imgs__   = " ".join(imgs),
                           __q__      = html.escape(q),
                           __more__   = f"search?q={urllib.parse.quote(q)}&num={bigger_num}&size={size}",
                           __tiny__   = f"search?q={urllib.parse.quote(q)}&size=100&num=200",
                           __small__  = f"search?q={urllib.parse.quote(q)}&size=200&num=100",
                           __medium__ = f"search?q={urllib.parse.quote(q)}&size=400&num=20",
                           __large__  = f"search?q={urllib.parse.quote(q)}&size=600&num=10",
                           __huge__  = f"search?q={urllib.parse.quote(q)}&size=800&num=10",
                           __num__  = num,
                           __size__ = size
                           )

@app.get("/",response_class=HTMLResponse)
def home():
    return """<html>
     <form action="search">Search: <input name="q" style="width:800px"></form>
    </html>"""

if __name__ == "__main__":
    uvicorn.run("rclip_server:app", host='0.0.0.0', port=8000, debug=True, reload=True)


# ~/proj/rclip/bin/rclip.sh -n dog -f | feh -f - t -g 1000x1000
