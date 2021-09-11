#!/usr/bin/env python

import argparse
import clip
import clip.model
import functools
import html
import io
import json
import numpy as np
import os
import pathlib
import PIL as pillow
import PIL.Image
import random
import requests.utils
import sqlite3
import string
import torch
import torch.nn
import urllib.parse
import uvicorn
import fastapi.responses
import seaborn
import matplotlib.colors
import matplotlib.cm

from dataclasses import dataclass,field
from typing import Optional
from fastapi import Cookie, FastAPI
from fastapi.responses import FileResponse,HTMLResponse
from PIL import Image
from starlette.requests import Request
from starlette.responses import StreamingResponse
from tqdm import tqdm
from typing import Callable, List, Tuple, cast
from typing import Iterable, List, NamedTuple, Tuple, TypedDict, cast

@dataclass
class ImageInfo:
    image_id        : int
    image_index     : int
    filename        : str
    thumbnail_url   : str = None
    detail_url      : str = None

class RClipServer:

    def __init__(self,rclip_db, model_name='ViT-B/32', device='cpu'):
        print(f"using {rclip_db}")
        self._db        = rclip_db
        self.device     = device
        self.model_name = model_name

        clip_model, clip_preprocess = clip.load(model_name,device)
        self.clip_model:      clip.model.CLIP                        = clip_model
        self.clip_preprocess: Callable[[Image.Image], torch.Tensor]  = clip_preprocess

        self.image_embeddings:numpy.ndarray         = None
        self.image_info:ImageInfo                   = None

        self.load_image_embeddings('')
        self.feature_minimums:np.ndarray = functools.reduce(lambda x,y: np.minimum(x,y), self.image_embeddings)
        self.feature_maximums:np.ndarray = functools.reduce(lambda x,y: np.maximum(x,y), self.image_embeddings)
        self.feature_ranges:np.ndarray   = self.feature_maximums - self.feature_minimums
        self.imgid_to_idx = dict([(ii.image_id,ii.image_index) for ii in self.image_info])
        print(f"found {len(self.image_info)} images")

    def download_image(self,url) -> PIL.Image.Image:

        headers = {'User-agent': # https://meta.wikimedia.org/wiki/User-Agent_policy
                   "rclip_server - similar image finder "+
                   "(http://152.67.254.195/ - still working on setting up contact info)"}
        resp = requests.get(url,headers=headers)
        if resp.headers['Content-Type'] not in ['image/jpeg','image/png','image/gif']:
            raise(Exception(f"unsupported {resp.headers['Content-Type']}"))
        stream = io.BytesIO(resp.content)
        img = pillow.Image.open(stream)
        return img
        
    def guess_user_intent(self,q):
        if re.match(r'^https?://',q):
            img = self.download_image(q)
            return self.get_image_embedding([img])

        if not q.startswith('{'):
            return self.get_text_embedding(q)

        data = json.loads(q)

        if img_id := data.get('image_id'):
            return np.asarray([self.image_embeddings[self.imgid_to_idx[img_id]]])

        if embedding := data.get('clip_embedding'):
            return np.asarray([embedding])

        if seed := data.get('random_img'):
            return np.asarray([random.choice(self.image_embeddings)])

        if seed := data.get('random_seed'):
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

    def get_text_embedding(self,words):
        with torch.no_grad():
            tokenized_text = clip.tokenize(words).to(self.device)
            text_encoded   = self.clip_model.encode_text(tokenized_text)
            text_encoded  /= text_encoded.norm(dim=-1, keepdim=True)
            return text_encoded.cpu().numpy()

    def get_image_embedding(self,images):
        with torch.no_grad():
            preprocessed = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)
            image_features = self.clip_model.encode_image(preprocessed)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()

    def image_info_from_id(self,img_id:int):
        return self.image_info[self.imgid_to_idx[img_id]]

    # Paraphrased From RClip
    def find_similar_images(self,desired_features: np.ndarray) -> List[Tuple[float, int]]:
        item_features = self.image_embeddings
        similarities = (desired_features @ item_features.T).squeeze(0).tolist()
        sorted_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        return sorted_similarities


    # Paraphrased From RClip
    def load_image_embeddings(self,directory: str) -> Tuple[List[str], np.ndarray]:
        ii: list[ImageInfo] = []
        image_embedding_list: list[np.ndarray] = []
        with sqlite3.connect(args.db) as con:
            con.row_factory = sqlite3.Row
            rclip_sql =  f'''
                SELECT *
                  FROM images 
                 WHERE filepath LIKE ? 
                   AND deleted IS NULL
            '''
            rows = con.execute(rclip_sql,('%',))
            cols = set([r[0] for r in rows.description])
            for (idx,row) in enumerate(rows):
                image_embedding_list.append(np.frombuffer(row['vector'], np.float32))
                if 'wikimedia_descr_url' in cols:
                    ii.append(ImageInfo(image_id        = row['id'],
                                        image_index     = idx,
                                        filename        = row['filepath'],
                                        thumbnail_url   = row['wikimedia_thumb_url'],
                                        detail_url      = row['wikimedia_descr_url']))
                else:
                    ii.append(ImageInfo(image_id        = row['id'],
                                        image_index     = idx,
                                        filename        = row['filepath']))
        self.image_embeddings = np.stack(image_embedding_list)
        self.image_info       = ii

    def censor(self,img_id):
        with sqlite3.connect(args.db) as con:
            con.row_factory = sqlite3.Row
            rclip_sql =  f'UPDATE images SET deleted=True where id = ?'
            return con.execute(rclip_sql,(img_id,))
        self.__init__(self._db)

    def reasonable_num(self,size:int):
        return (size < 100 and 360 
                or size < 200 and 180
                or size < 400 and 72
                or size < 600 and 24
                or size < 800 and 12
                or 6)

    # Show candidate words for picture
    
    def find_best_matches(self,desired_embedding: np.ndarray, all_embeddings) -> list[tuple[float, int]]:
        similarities = (desired_embedding @ all_embeddings.T).squeeze(0).tolist()
        sorted_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        return sorted_similarities

    def load_word_embeddings(self,word_embedding_db):
        word_features: list[np.ndarray] = []
        words:         list[str] = []
        with sqlite3.connect(word_embedding_db) as con:
            con.row_factory = sqlite3.Row
            sql =  f'''
                    SELECT *
                      FROM words 
                      WHERE lower(words) = words
           '''
            rows = con.execute(sql,())
        for row in rows:
            word_features.append(np.frombuffer(row['vector'], np.float32))
            words.append(row['words'])
        self.word_embeddings = np.stack(word_features)
        self.words           = words
        self.word_to_idx     = dict([(word,idx) for idx,word in enumerate(words)])

    def best_words(self,desired_embedding):
        best_matches = self.find_best_matches(desired_embedding,self.word_embeddings)
        return [(idx,self.words[idx],score) for idx,score in best_matches]

    def guess_phrase_embedding(self,words):
        embeddings = [self.word_embeddings[self.word_to_idx[word]] for word in words]
        combined   = functools.reduce(lambda x,y: x + y,embeddings)
        normalized = combined / np.linalg.norm(combined)
        return normalized

    def guess_phrase_score(self,desired_embedding,words):
        pe = self.guess_phrase_embedding(words)
        return (desired_embedding @ pe.T).squeeze(0).tolist()

    # slow - and the estimate above is pretty close.
    def calculate_phrase_score(self,desired_embedding,words):
        pe = self.get_text_embedding(words)
        return (desired_embedding @ pe.T).squeeze(0).tolist()[0]

    def best_phrases(self,desired_embedding):
        best_words    = self.best_words(desired_embedding)[0:50]
        num_per_group = 1000
        candidate_phrases   = ([[w[1] for w in random.sample(best_words, 2)] for i in range(num_per_group)] +
                               [[w[1] for w in random.sample(best_words, 3)] for i in range(num_per_group)] +
                               [[w[1] for w in random.sample(best_words, 4)] for i in range(num_per_group)]
                               )
        candidate_phrases_with_scores = [(" ".join(words),
                                          self.guess_phrase_score(desired_embedding,words))
                                          for words in candidate_phrases
                                        ]
        sorted_phrases = sorted(candidate_phrases_with_scores,key=lambda p:p[1],reverse=True)
        #candidate_word_vecs           = [self.word_embeddings[idx] for idx,word,score in best_words]
        #first_word_vec                = candidate_word_vecs[0]
        #candidate_phrase_vecs         = [(v + first_word_vec) / np.linalg.norm(v + first_word_vec) for v in candidate_word_vecs]
        #candidate_phrase_scores       = [(desired_embedding @ cv.T).squeeze(0).tolist() for cv in candidate_phrase_vecs]
        #candidate_phrases_with_scores = [(bw[0],bw[1],bw[2],s) for bw,s in zip(best_words,candidate_phrase_scores)]
        return sorted_phrases[:100]

################################################################################
# Create FastAPI server
#   Uses argparse and environment variables because uvicorn swallows args
################################################################################

default_db                = (os.environ.get("CLIP_DB") or
                             pathlib.Path.home() / '.local/share/rclip/db.sqlite3')
parser                    = argparse.ArgumentParser()
parser.add_argument('--db','-d',default=str(default_db),help="path to rclip's database")
args,unk                  = parser.parse_known_args()
if os.path.exists(args.db):
  rclip_server            = RClipServer(args.db)
  rclip_server.load_word_embeddings('words.sqlite3')
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
    features = rclip_server.get_text_embedding(q)
    if p:
      plus_features = rclip_server.get_text_embedding(p)
      features = features + plus_features
    if m:
      minus_features = rclip_server.get_text_embedding(m)
      features = features - minus_features
    combined_features = features
    print(features)
    results = rclip_server.find_similar_images(combined_features)
    print(results[0:10])
    return make_html(results,q,size,num,debug_features = combined_features)

@app.get("/censor", response_class=HTMLResponse)
async def censor_endpoint(img_id:int):
    rclip_server.censor(img_id)
    return(f"<html>Ok, {path} is censored</html>")

@app.get("/reload", response_class=HTMLResponse)
async def reload():
    rclip_server.__init__(rclip_server._db)
    return fastapi.responses.RedirectResponse('/')

@app.get("/img/{img_id}")
async def img(img_id:int):
  ii = rclip_server.image_info_from_id(img_id)
  if ii.detail_url:
        return fastapi.responses.RedirectResponse(ii.detail_url)
  hdrs = {'Cache-Control': 'public, max-age=172800'}
  return FileResponse(ii.filename, headers=hdrs)

import re
@app.get("/thm/{img_id}")
async def thm(img_id:int, size:Optional[int]=400):
  ii = rclip_server.image_info_from_id(img_id)
  if ii.thumbnail_url:
    thm_url = ii.thumbnail_url
    thm_url = re.sub(r'/600px-',f'/{size}px-',thm_url)
    return fastapi.responses.RedirectResponse(thm_url)
  img = Image.open(ii.filename)
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
    clip_embedding = rclip_server.get_image_embedding([img])
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
    scores_with_imgids = [(score,rclip_server.image_info[idx]) for idx,score in sims[:num]]
    debug=True
    imgs = [f"""
             <div>
                <a href="/img/{ii.image_id}" target="_blank"><img src="/thm/{ii.image_id}?size={size}"></a>
                <br>
                {str(int(100*s))+'% similarity'}
                <a href='/search?q={urllib.parse.quote(json.dumps({"image_id":ii.image_id,'path':ii.filename}))}'>more like this</a>
                <!-- <a href="/censor?img_id={ii.image_id}">censor this</a>-->
                {debug and "" or "<!--"} | <a href="/mut?q={ii.image_id}">unlike this</a> {debug and "" or "-->"}
             </div>
             """ 
            for s,ii in scores_with_imgids
            ]
    tmpl = string.Template("""<html>
       <title>$__title__</title>
       <style>
          body {background-color: #444; width: 100%; margin: 0px; 
                color:white;
                font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", "Ubuntu", "Helvetica Neue", sans-serif;
          }
          form {margin:0px}
          #header,#footer {background-color: #444; padding: 20px 20px 10px 20px; }
          div.images {padding:2px;; margin:auto;}
          .images div{display:inline-block; width:${__size__}px; margin:auto;}
          .images img{display:inline-block; max-width:${__size__}px;max-height:${__hsize__}px}
          #q {width:99%}
          #lq {font-size: 20pt}
          #sizes,.images {font-size: 10pt}
          a {text-decoration: none; color: #ccccff}
          a:hover {text-decoration: underline;}
          th {text-align:left}
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
            <td width="10%"><label for="q" id="lq"><a href="/">Search:</a></label></td>
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
    cmap       = seaborn.color_palette('icefire',as_cmap=True)
    norm       = matplotlib.colors.Normalize(vmin=0, vmax=1)
    scalar_map = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
    def get_color(f):
      rgba = scalar_map.to_rgba(f)
      return "".join([f'{int(255*x):02x}' for x in rgba[0:3]])
    if debug_features is not None:
        debug_txt += "<div style='width:100%'><div style='margin:auto'>"

        clip_vec_as_json = json.dumps({"clip_embedding":debug_features.flatten().tolist()})
        debug_txt += "<h2>CLIP Embedding</h2>"
        debug_txt += f"<a href=search?q={urllib.parse.quote(clip_vec_as_json)}>CLIP embedding</a>: (red = above the mean for this dataset; blue = below the mean for this dataset)"
        debug_txt += "<table><tr>"
        normalized_debug_features = (debug_features - rclip_server.feature_minimums) / rclip_server.feature_ranges
        zipped_features = zip(debug_features.flatten(),normalized_debug_features.flatten())
        for idx,(df,nf) in enumerate(zipped_features):
            debug_txt += f"""<td style="background:#{get_color(nf)}">{float(df):0.2g}</td>"""
            if idx % 16 == 15: debug_txt += "</tr><tr>" 
        debug_txt += "</table>"

        debug_txt += "<h2>Similar words and phrases</h2>"
        best_words = rclip_server.best_words(debug_features)[0:50]
        debug_txt += "<table style='width:40%; float:left'><tr><th>word</th><th>score</th></tr>\n"
        for idx,word,score in best_words:
            u = f'/search?q={urllib.parse.quote(word)}'
            debug_txt += f"<tr><td><a href='{u}'>{html.escape(word)}</a></td><td>{int(score*100)}%</td></tr>"
        debug_txt += "</table>"

        best_phrases = rclip_server.best_phrases(debug_features)[0:50]
        debug_txt += "<table style='width:40%; float:left'><tr><th>phrases</th><th>est. score</th></tr>\n"
        for phrase,score in best_phrases:
            u = f'/search?q={urllib.parse.quote(phrase)}'
            debug_txt += f"<tr><td><a href='{u}'>{html.escape(phrase)}</a></td><td>{int(score*100)}%</td></tr>"
        debug_txt += "</table>"
        debug_txt += "</div></div>"

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
                           __hsize__     = int(size*3/4),
                           __debug_txt__ = debug_txt
                           )

###############################################################################
# Launch the server
###############################################################################

if __name__ == "__main__":
  # uvicorn.run("rclip_server:app", host='0.0.0.0', port=8000, debug=True, reload=True)
  print("Best run using a command like like:")
  print(" uvicorn rclip_server:app")
  print("to launch an instance using the user's default rclip database, or")
  print(" env CLIP_DB=wikimedia_quality.sqlite3 uvicorn rclip_server:app --reload")
  print("to run a dev instance specifying a different index")

# Debug the rclip index using commands like:
#
#    ~/proj/rclip/bin/rclip.sh -n dog -f | feh -f - t -g 1000x1000
#
# If using nginx, add the following config to allow the large GET reqests with clip embedding paramaters:
#
#   large_client_header_buffers 16 16k;
