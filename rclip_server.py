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
import random
import requests.utils
import sqlite3
import string
import torch
import torch.nn
import urllib.parse
import uvicorn
import fastapi.responses

from typing import Optional
from fastapi import Cookie, FastAPI
from fastapi.responses import FileResponse,HTMLResponse
from PIL import Image
from starlette.requests import Request
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

#model = Model()

class RClipServer:

    def __init__(self,rclip_db):
        self._db    = rclip_db
        self._model = Model()
        self.filepaths:        List[str]            = []
        self.wikimedia_info:   List[Tuple(str,str)] = []
        self.global_features:  List[np.ndarray]     = []
        self.img_ids:          List[int]            = []
        self.filepaths,self.global_features,self.img_ids,self.wikimedia_info = self._get_features('')
        self.feature_minimums:np.ndarray = functools.reduce(lambda x,y: np.minimum(x,y), self.global_features)
        self.feature_maximums:np.ndarray = functools.reduce(lambda x,y: np.maximum(x,y), self.global_features)
        self.feature_ranges:np.ndarray   = self.feature_maximums - self.feature_minimums
        self.idx_to_imgid = dict(enumerate(self.img_ids))
        self.imgid_to_idx = dict([(y,x) for x,y in enumerate(self.img_ids)])
        print(f"found {len(self.img_ids)} images")

    def guess_user_intent(self,q):
        if not q.startswith('{'):
            return self.compute_text_features(q)

        data = json.loads(q)

        if img_id := data.get('image_id'):
            return np.asarray([self.global_features[self.imgid_to_idx[img_id]]])
            img_path = self.get_path_from_id(img_id)
            img     = Image.open(img_path)
            return self.compute_image_features([img])

        if embedding := data.get('clip_embedding'):
            return np.asarray([embedding])

        if seed := data.get('random_img'):
            return np.asarray([random.choice(self.global_features)])

        if seed := data.get('random_seed'):

            #rng = np.random.default_rng(seed)
            #rnd_features = rng.random(512) * rclip_server.feature_ranges + rclip_server.feature_minimums
            #rnd_features /= np.linalg.norm(rnd_features)

            random.seed(seed)
            def make_rand_vector(dims):
                """
                   https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space 
                """
                vec = [random.gauss(0, 1) for i in range(dims)]
                mag = sum(x**2 for x in vec) ** .5
                return [x/mag for x in vec]
            rnd_features = make_rand_vector(512)

            return np.asarray([rnd_features])

    def compute_image_features(self, images: List[Image.Image]) -> np.ndarray:
        return self._model.compute_image_features(images)

    def compute_text_features(self, text:str) -> np.ndarray:
        return self._model.compute_text_features(text)

    def get_path_from_id(self,img_id:int):
        return self.filepaths[self.imgid_to_idx[img_id]]

    def get_wikimedia_info_from_id(self,img_id:int):
        return self.wikimedia_info[self.imgid_to_idx[img_id]]

    # Paraphrased From RClip
    def find_similar_images(self,desired_features: np.ndarray) -> List[Tuple[float, int]]:
        item_features = self.global_features
        similarities = (desired_features @ item_features.T).squeeze(0).tolist()
        sorted_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        return sorted_similarities

    # Paraphrased From RClip
    def load_vecs(self,path):
        with sqlite3.connect(args.db) as con:
            con.row_factory = sqlite3.Row
            rclip_sql =  f'''
                SELECT *
                  FROM images 
                 WHERE filepath LIKE ? 
                   AND deleted IS NULL
            '''
            return con.execute(rclip_sql,(path + '%',))

    # Paraphrased From RClip
    def _get_features(self,directory: str) -> Tuple[List[str], np.ndarray]:
        filepaths: List[str] = []
        features: List[np.ndarray] = []
        img_ids: List[int] = []
        wikimedia_info: List[Tuple(str,str)] = []
        print("here")
        rows = self.load_vecs('%')
        cols = set([r[0] for r in rows.description])
        for image in rows:
          filepaths.append(image['filepath'])
          features.append(np.frombuffer(image['vector'], np.float32))
          img_ids.append(image['id'])
          if 'wikimedia_descr_url' in cols:
              wikimedia_info.append((image['wikimedia_descr_url'],
                                     image['wikimedia_thumb_url']))
          else:
              wikimedia_info.append(None)
        return filepaths, np.stack(features), img_ids, wikimedia_info

    def censor(self,path):
        with sqlite3.connect(args.db) as con:
            con.row_factory = sqlite3.Row
            rclip_sql =  f'UPDATE images SET deleted=True where filepath = ?'
            return con.execute(rclip_sql,(path,))
        self.__init__(self._db)

    def reasonable_num(self,size:int):
        return (size < 100 and 360 
                or size < 200 and 180
                or size < 400 and 72
                or size < 600 and 24
                or size < 800 and 12
                or 6)

################################################################################
# Create FastAPI server
################################################################################

parser                    = argparse.ArgumentParser()
default_db                = pathlib.Path.home() / '.local/share/rclip/db.sqlite3'
parser.add_argument('--db','-d',default=str(default_db),help="path to rclip's database")
args,unk                  = parser.parse_known_args()
rclip_server              = RClipServer(args.db)
app                       = FastAPI()

###############################################################################
# HTTP Endpoints
###############################################################################

@app.get("/",response_class=HTMLResponse)
def home():
    return make_html([],'',400,20,debug_features = None)

@app.get("/search", response_class=HTMLResponse)
async def search(q:str, num:Optional[int] = None, size:int=Cookie(400)):
    desired_features = rclip_server.guess_user_intent(q)
    results = rclip_server.find_similar_images(desired_features)
    return make_html(results,q,size,num,debug_features = desired_features)

@app.get("/opposite", response_class=HTMLResponse)
async def opposite(q:str, num:Optional[int] = None, size:int=Cookie(400)):
    desired_features = rclip_server.guess_user_intent(q)
    results = rclip_server.find_similar_images(desired_features * -1)
    return make_html(results,q,size,num,debug_features = desired_features)

@app.get("/conceptmap", response_class=HTMLResponse)
async def conceptmap(q:str, m:str=None, p:str=None,num:int = 36, size:int=400, debug:bool=False):
    features = rclip_server.compute_text_features(q)
    if p:
      plus_features = rclip_server.compute_text_features(p)
      features = features + plus_features
    if m:
      minus_features = rclip_server.compute_text_features(m)
      features = features - minus_features
    combined_features = features
    print(features)
    results = rclip_server.find_similar_images(combined_features)
    print(results[0:10])
    return make_html(results,q,size,num,debug_features = combined_features)

@app.get("/censor", response_class=HTMLResponse)
async def censor_endpoint(img_id:int):
    path = rclip_server.get_path_from_id(img_id)
    rclip_server.censor(path)
    return(f"<html>Ok, {path} is censored</html>")

@app.get("/reload", response_class=HTMLResponse)
async def reload():
    rclip_server.__init__(rclip_server._db)
    return fastapi.responses.RedirectResponse('/')

@app.get("/img/{img_id}")
async def img(img_id:int):
    if wikimedia_info := rclip_server.get_wikimedia_info_from_id(img_id):
        return fastapi.responses.RedirectResponse(wikimedia_info[0])
    img_path = rclip_server.get_path_from_id(img_id)
    hdrs = {'Cache-Control': 'public, max-age=172800'}
    return FileResponse(img_path, headers=hdrs)

import re
@app.get("/thm/{img_id}")
async def thm(img_id:int, size:Optional[int]=400):
    img_path = rclip_server.get_path_from_id(img_id)
    print(img_path)
    if wikimedia_info := rclip_server.get_wikimedia_info_from_id(img_id):
        thm_url = wikimedia_info[1]
        thm_url = re.sub(r'/600px-',f'/{size}px-',thm_url)
        return fastapi.responses.RedirectResponse(thm_url)
    img = Image.open(img_path)
    thm = img.thumbnail((size,3*size/4))
    buf = io.BytesIO()
    if img.mode != 'RGB': img = img.convert('RGB')
    img.save(buf,format="jpeg")
    buf.seek(0)
    hdrs = {'Cache-Control': 'public, max-age=172800'}
    return StreamingResponse(buf,media_type="image/jpeg", headers=hdrs)

@app.get("/info/{img_id}")
async def info(img_id:int):
    img_path = rclip_server.get_path_from_id(img_id)
    img     = Image.open(img_path)
    clip_embedding = rclip_server.compute_image_features([img])
    info = {'path':img_path,'embedding':clip_embedding.tolist()}
    return fastapi.responses.JSONResponse(content=info)


###############################################################################
# Minimal HTML template
###############################################################################

def dedup_sims(similarities):
    seen = set()
    return [seen.add(s[1]) or s for s in similarities if s[1] not in seen and not seen.add(s[1])]

def make_html(similarities,q,size,num,debug_features=None,debug=False):
    num = num or rclip_server.reasonable_num(size)
    sims = dedup_sims(similarities[:num*2])
    print(sims)
    scores_with_imgids = [(score,rclip_server.idx_to_imgid[idx]) for idx,score in sims[:num]]
    debug=True
    imgs = [f"""
             <div style="">
                <a href="/img/{img_id}" target="_blank"><img src="/thm/{img_id}?size={size}" style='max-width:{size}px; max-height:{size}px'></a>
                <br>
                {str(int(100*s))+'% similarity'}
                <a href='/search?q={urllib.parse.quote(json.dumps({"image_id":img_id,'path':rclip_server.get_path_from_id(img_id)}))}'>more like this</a>
                <!-- <a href="/censor?img_id={img_id}">censor this</a>-->
                {debug and "" or "<!--"} | <a href="/mut?q={img_id}">unlike this</a> {debug and "" or "-->"}
             </div>
             """ 
            for s,img_id in scores_with_imgids
            ]
    tmpl = string.Template("""<html>
        <title>$__title__</title>
       <style>
          body {background-color: #ccc; width: 100%; font-family:Ariel}
          form {margin:0px}
          #header,#footer {background-color: #888; padding: 10px 20px 10px 20px; }
          .images div{display:inline-block; width:${__size__}px}
          #q {width:100%}
          #lq {font-size: 20pt}
          #sizes,.images {font-size: 10pt}
          a:link {text-decoration: none}
          a:hover {text-decoration: underline}
       </style>
       <script>
            function setCookie(cname, cvalue, exdays) {
              const d = new Date();
              d.setTime(d.getTime() + (exdays*24*60*60*1000));
              let expires = "expires="+ d.toUTCString();
              document.cookie = cname + "=" + cvalue + ";" + expires + ";path=/;SameSite=Lax";
            }
            function set_size(s) {
                setCookie('size',s,2);
                location.reload(); 
            }
            /*
           fetch("https://ipinfo.io/json")
               .then(function (response) {
                   return response.json();
               })
               .then(function (myJson) {
                   console.log(myJson.ip);
               })
               .catch(function (error) {
                   console.log("Error: " + error);
              });
            */
       </script>

       <div id="header">
       <form action="search">
           <table width="100%"><tr>
            <td width="10%"><label for="q" id="lq"><a href="/" style="color:black; text-decoration: none">Search:</a></label></td>
            <td width="80%"><input name="q" id='q' value="$__q__" style="width: width:800px"></td>
            <td width="10%"><input type="submit" value="Go"></td>
           </tr><tr><td></td>
            <td id="sizes">
            <div style="float:right"><a href="$__rnd__">random</a></div>
            <div id="sizelinks" style="width:50%">
             <a href="javascript:set_size(100)">tiny</a>
             <a href="javascript:set_size(200)">small</a>
             <a href="javascript:set_size(400)">medium</a>
             <a href="javascript:set_size(600)">large</a>
             <a href="javascript:set_size(800)">huge</a>
             </div>
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
        clip_vec_as_json = json.dumps({"clip_embedding":debug_features.flatten().tolist()})
        debug_txt += f"<a href=search?q={urllib.parse.quote(clip_vec_as_json)}>CLIP embedding</a>:"
        debug_txt += "<table><tr>"
        normalized_debug_features = 255 * (debug_features - rclip_server.feature_minimums) / rclip_server.feature_ranges
        zipped_features = zip(debug_features.flatten(),normalized_debug_features.flatten())
        for idx,(df,nf) in enumerate(zipped_features):
            clr = nf > 255 and 255 or int(nf) < 0 and 0 or int(nf)
            debug_txt += f"""<td style="background:#{clr:02x}{clr:02x}ff">{float(df):0.2g}</td>"""
            if idx % 16 == 15: debug_txt += "</tr><tr>" 
        debug_txt += "</table>"
    bigger_num = (num > 100) and 1000 or num * 10
    rnd_param = json.dumps({'random_seed':random.randint(0,10000)})
    return tmpl.substitute(__imgs__      = " ".join(imgs),
                           __q__         = html.escape(q),
                           __title__     = f"rclip_server {html.escape(q)}",
                           __opposite__  = f"opposite?q={urllib.parse.quote(q)}&num={num}",
                           __more__      = f"search?q={urllib.parse.quote(q)}&num={bigger_num}",
                           __rnd__       = f"search?q={urllib.parse.quote(rnd_param)}",
                           __num__       = num,
                           __size__      = size,
                           __debug_txt__ = debug_txt
                           )

###############################################################################
# Launch the server
###############################################################################

if __name__ == "__main__":
    uvicorn.run("rclip_server:app", host='0.0.0.0', port=8000, debug=True, reload=True)

# Debug with:
# ~/proj/rclip/bin/rclip.sh -n dog -f | feh -f - t -g 1000x1000
