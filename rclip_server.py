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
import re
import requests.utils
import sqlite3
import string
import sys
import time
import torch
import torch.nn
import urllib.parse
import uvicorn
import fastapi.responses
import seaborn
import matplotlib.colors
import matplotlib.cm
import pyparsing as pp

from collections.abc import Iterable
from dataclasses import dataclass,field
from typing import Optional
from fastapi import Cookie, FastAPI
from fastapi.responses import FileResponse,HTMLResponse
from starlette.responses import StreamingResponse
from tqdm import tqdm

@dataclass
class ImageInfo:
    image_id        : int
    image_index     : int
    filename        : str
    thumbnail_url   : str = None
    detail_url      : str = None

class RClipServer:

    def __init__(self, rclip_db:str, model_name:str='ViT-B/32', device:str='cpu') -> None:
        self.rclip_db: str  = rclip_db
        self.device:str     = device
        self.model_name:str = model_name

        clip_model, clip_preprocess = clip.load(model_name,device)
        self.clip_model: clip.model.CLIP                                 = clip_model
        self.clip_preprocess: callable[[PIL.Image.Image], torch.Tensor]  = clip_preprocess

        image_embeddings,image_info = self.load_image_embeddings('')
        self.image_embeddings:np.ndarray = image_embeddings
        self.image_info:list[ImageInfo]  = image_info
        self.imgid_to_idx                = dict([(ii.image_id,ii.image_index) for ii in image_info])
        self.feature_minimums:np.ndarray = functools.reduce(lambda x,y: np.minimum(x,y), image_embeddings)
        self.feature_maximums:np.ndarray = functools.reduce(lambda x,y: np.maximum(x,y), image_embeddings)
        self.feature_ranges:np.ndarray   = self.feature_maximums - self.feature_minimums
        self.parser:pp.ParserElement     = self.get_parser()

        if os.path.exists('words.sqlite3'):
            self.load_word_embeddings('words.sqlite3')

        print(f"found {len(self.image_info)} images")

    def download_image(self,url) -> PIL.Image.Image:
        headers = {'User-agent': # https://meta.wikimedia.org/wiki/User-Agent_policy
                   "rclip_server - similar image finder - "+
                   "https://github.com/ramayer/rclip-server/"}
        # resp = requests.get(url,headers=headers)
        # if resp.headers['Content-Type'] not in ['image/jpeg','image/png','image/gif']:
        #     raise(Exception(f"unsupported {resp.headers['Content-Type']}"))
        # stream = io.BytesIO(resp.content)
        # img = pillow.Image.open(stream)
        img = pillow.Image.open(requests.get(url, headers=headers, stream=True, timeout=60).raw)
        return img

    def get_parser(self) -> pp.ParserElement:
        # using pre-release pyparsing==3.0.0rc1 , so I don't need to change APIs later
        sign = pp.Opt(
                    pp.Group(
                        pp.one_of('+ -') +
                        pp.Opt(pp.pyparsing_common.number.copy(),'1')
                    )
              ,['+','1'])
        #word  = pp.Word(pp.alphanums,exclude_chars='([{}])') # fails on hyphenated words
        #word  = pp.Word(pp.alphanums,pp.printables,exclude_chars='([{}])') # fails on unicode
        word  = pp.Word(pp.unicode.alphanums,pp.unicode.printables,exclude_chars='([{}])') # slow
        words = pp.OneOrMore(word)
        enclosed        = pp.Forward()
        quoted_string   = pp.QuotedString('"')
        nested_parens   = pp.nestedExpr('(', ')', content=enclosed)
        nested_brackets = pp.nestedExpr('[', ']', content=enclosed)
        nested_braces   = pp.nestedExpr('{', '}', content=enclosed)
        enclosed << pp.OneOrMore((pp.Regex(r"[^{(\[\])}]+") | nested_parens | nested_brackets | nested_braces | quoted_string))
        expr = (sign + 
                pp.original_text_for((quoted_string | nested_parens | nested_braces | words))
                )
        return expr

    def guess_user_intent(self,q) -> np.ndarray:
        """Support complex queries like
           skiing +summer -winter -(winter sports) +(summer sports) +{img:1234} +https://example.com/img.jpg


           * TODO 
              Consider if adding and subtracting vectors is really what we want here.

              Perhaps we should think more in terms of rotating in the direction of a different term.
              See also:
                  https://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/

                  "If you want to interpolate between two random
                  without leaving the soap bubble you should instead
                  interpolate in in polar coordinates ... For a more
                  robust spherical interpolation scheme you might want
                  to opt for something like SLERP."

             Some syntax for "start a zerbra, and rotate 20 dgegrees
             in the direction of horse" might be interesting.

        """
        parser = self.parser
        parsed = parser.search_string(q)
        embeddings = []
        for (operator,magnitude),terms in parsed:
            if len(terms)>2 and terms[0] == '(' and terms[-1] == ')': terms=terms[1:-1]
            #print(operator,magnitude,terms)
            e = self.guess_user_intent_element(terms) * float(magnitude) * float(operator+'1')
            embeddings.append(e)
        if len(embeddings) == 0:
            return None
        result = functools.reduce(lambda x,y: x+y, embeddings)
        result /= np.linalg.norm(result)
        return result

    @functools.lru_cache
    def guess_user_intent_element(self,q) -> np.ndarray:
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
            def rand_ndim_unit_vector(dims):
                """
                   https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space 
                """
                vec = [random.gauss(0, 1) for i in range(dims)]
                mag = sum(x**2 for x in vec) ** .5
                return [x/mag for x in vec]
            rnd_features = rand_ndim_unit_vector(512)
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
    def find_similar_images(self,desired_features: np.ndarray) -> list[tuple[float, int]]:
        item_features = self.image_embeddings
        similarities = (desired_features @ item_features.T).squeeze(0).tolist()
        sorted_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        return sorted_similarities

    # Paraphrased From RClip
    def load_image_embeddings(self,directory: str) -> tuple[list[str], np.ndarray]:
        ii: list[ImageInfo] = []
        image_embedding_list: list[np.ndarray] = []
        with sqlite3.connect(args.db) as con:
            con.row_factory = sqlite3.Row
            rclip_sql =  f'''
                SELECT *
                  FROM images 
                 WHERE filepath LIKE ? 
                   AND (deleted IS null or deleted = false)
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
        #self.image_embeddings = np.stack(image_embedding_list)
        #self.image_info       = ii
        return np.stack(image_embedding_list),ii

    def censor(self,img_id):
        with sqlite3.connect(self.rclip_db) as con:
            con.row_factory = sqlite3.Row
            rclip_sql =  f'UPDATE images SET deleted=True where id = ?'
            return con.execute(rclip_sql,(img_id,))
        self.__init__(self.rclip_db)

    def dedup_sqlite(self):
        """ Set the deleted flag for any image that has the exact same embedding as another image """
        dedup_sql = '''
            WITH a AS (
               SELECT count(*),min(id) AS min_id FROM images WHERE deleted IS NULL GROUP BY vector HAVING count(*) > 1
           ) UPDATE images SET deleted=true 
              WHERE vector IN (SELECT vector FROM images WHERE id IN (SELECT min_id FROM a)) 
                AND id NOT IN (SELECT min_id FROM a)
        '''

    def find_best_matches(self,desired_embedding: np.ndarray, all_embeddings) -> list[tuple[float, int]]:
        """ Compute dot produts of all candidate resutls with the desired result, and return a sorted list """
        similarities = (desired_embedding @ all_embeddings.T).squeeze(0).tolist()
        sorted_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        return sorted_similarities

    def visualize_clip_embedding(self,desired_embedding):
        cmap       = seaborn.color_palette('icefire',as_cmap=True)
        norm       = matplotlib.colors.Normalize(vmin=0, vmax=1)
        scalar_map = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
        def get_color(f):
          rgba = scalar_map.to_rgba(f)
          return "".join([f'{int(255*x):02x}' for x in rgba[0:3]])

        html_result = "<div style='margin:auto; display:table;'>"
        clip_vec_as_json = json.dumps({"clip_embedding":desired_embedding.flatten().tolist()})
        html_result += "<h2>CLIP Embedding</h2>"
        html_result += (f"<a href=search?q={urllib.parse.quote(clip_vec_as_json)}>CLIP embedding</a>:<br>"+
                        "* red = above the mean for this dataset<br> * blue = below the mean for this dataset")
        html_result += "<table style='font-size:7pt'><tr>"
        normalized_desired_embedding = (desired_embedding - rclip_server.feature_minimums) / rclip_server.feature_ranges
        zipped_features = zip(desired_embedding.flatten(),normalized_desired_embedding.flatten())
        for idx,(df,nf) in enumerate(zipped_features):
            html_result += f"""<td style="background:#{get_color(nf)}">{float(df):0.2g}</td>"""
            if idx % 8 == 7: html_result += "</tr><tr>" 
        html_result += "</table></div>"
        return html_result

    def copyright_message(self):
        if re.search(r'wiki',self.rclip_db):
            return '''             
            <small>Images in this demo are from Wikimedia Commons,
            available under various different licenses specified on
            their description page.  Click on an image to see its
            specific license.<br>
            Source code for this project is 
            <a href="https://github.com/ramayer/rclip-server">available
            on github</a>.</small>
            '''
        else:
            return self.rclip_db

    ############################################################################
    ## Optionally load precomputed clip embeddings of words to display
    ############################################################################
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

    # approximate, but fast
    def guess_phrase_score(self,desired_embedding,words):
        pe = self.guess_phrase_embedding(words)
        return (desired_embedding @ pe.T).squeeze(0).tolist()

    # slow - and the estimate above is pretty close.
    def calculate_phrase_score(self,desired_embedding,words):
        pe = self.get_text_embedding(words)
        return (desired_embedding @ pe.T).squeeze(0).tolist()[0]

    def best_phrases(self,desired_embedding):
        best_words    = self.best_words(desired_embedding)[0:200]
        num_per_group = 1000
        candidate_phrases   = ([[w[1] for w in random.sample(best_words, 2)] for i in range(num_per_group)] +
                               [[w[1] for w in random.sample(best_words, 3)] for i in range(num_per_group)] +
                               [[w[1] for w in random.sample(best_words, 4)] for i in range(num_per_group)]
                               )
        candidate_phrases_with_scores = [(" ".join(words),
                                          self.guess_phrase_score(desired_embedding,words))
                                          for words in candidate_phrases
                                        ]
        sorted_phrases = sorted(candidate_phrases_with_scores,key=lambda p:p[1],reverse=True)[:100]
        return sorted_phrases[:100]

################################################################################
# Load a rclip database
# Also use environment variables, because uvicorn swallows args
################################################################################

default_db                = (os.environ.get("CLIP_DB") or
                             pathlib.Path.home() / '.local/share/rclip/db.sqlite3')
parser                    = argparse.ArgumentParser()
parser.add_argument('--db','-d',default=str(default_db),help="path to rclip's database")
args,unk                  = parser.parse_known_args()
if os.path.exists(args.db):
  rclip_server            = RClipServer(args.db)


################################################################################
# Create FastAPI server
#   Uses argparse and environment variables because uvicorn swallows args
################################################################################

app                       = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# HTTP Endpoints

@app.get("/",response_class=HTMLResponse)
def home():
  hdrs = {'Cache-Control': 'public, max-age=3600'}
  return FileResponse('./assets/rclip_server.html', headers=hdrs)

@app.get("/search", response_class=HTMLResponse)
async def search(q:str, num:Optional[int] = None, size:int=Cookie(400)):
    hdrs = {'Cache-Control': 'public, max-age=3600'}
    return FileResponse('./assets/rclip_server.html', headers=hdrs)

@app.get("/search_api")
async def search_api(q:str, num:Optional[int] = None):
    desired_features = rclip_server.guess_user_intent(q)
    if desired_features is None: return []
    results = rclip_server.find_similar_images(desired_features)
    top_results = results[0:num or 12]
    r = [(rclip_server.image_info[idx].image_id,score) for idx,score in top_results]
    return r

@app.get("/similar_words")
async def similar_words(q:str, num:Optional[int] = None):
    desired_features = rclip_server.guess_user_intent(q)
    if desired_features is None: return []
    similar_words    = rclip_server.best_words(desired_features)[0:50]
    similar_phrases  = rclip_server.best_phrases(desired_features)[0:50]
    result = {'similar_words':similar_words,'similar_phrases':similar_phrases}
    return result

@app.get("/clip_embedding")
async def clip_embedding(q:str, num:Optional[int] = None):
    desired_features = rclip_server.guess_user_intent(q)
    embedding_list = desired_features.squeeze().tolist()
    return({'clip_embedding':embedding_list})

@app.get("/clip_text_embedding")
async def clip_text_embedding(q:str):
    desired_features = rclip_server.get_text_embedding(q)
    embedding_list = desired_features.squeeze().tolist()
    return({'clip_text_embedding':embedding_list})

@app.get("/visualize_clip_embedding")
async def visualize_clip_embedding_api(q:str, num:Optional[int] = None):
    desired_features = rclip_server.guess_user_intent(q)
    if desired_features is None: return ''
    html_frag = rclip_server.visualize_clip_embedding(desired_features)
    return({'clip_embedding':html_frag})

@app.get("/censor/{img_id}")
async def censor_endpoint(img_id:int,censorship_key:str):
    if censorship_key != os.environ.get("RCLIP_SERVER_CENSORSHIP_KEY"):
        return({"error":"censorship key didn't match"})
    rclip_server.censor(img_id)
    return({"msg":f"Ok. {img_id} is now censored"})

@app.get("/reload", response_class=HTMLResponse)
async def reload():
    rclip_server.__init__(rclip_server.rclip_db)
    return fastapi.responses.RedirectResponse('/')

@app.get("/js/vue.global.prod.js")
async def vue_js():
  hdrs = {'Cache-Control': 'public, max-age=172800'}
  return FileResponse('./assets/vue@3.2.11/vue.global.prod.js', headers=hdrs)

@app.get("/img/{img_id}")
async def img(img_id:int):
  ii = rclip_server.image_info_from_id(img_id)
  if ii.detail_url:
        return fastapi.responses.RedirectResponse(ii.detail_url)
  hdrs = {'Cache-Control': 'public, max-age=172800'}
  return FileResponse(ii.filename, headers=hdrs)

@app.get("/thm/{img_id}")
async def thm(img_id:int, size:Optional[int]=400):
  hdrs = {'Cache-Control': 'public, max-age=172800'}
  if img_id == -1:
     svg = f'''<svg version="1.1" width="{size}" height="{int(size*3/4)}" xmlns="http://www.w3.org/2000/svg">
              <!--<rect width="100%" height="100%" fill="#333" /> -->
              <circle cx="50%" cy="50%" r="25%" fill="#222"/>
              </svg>'''
     buf = io.BytesIO(bytes(svg,'utf-8'))
     buf.seek(0)
     return StreamingResponse(buf,media_type="image/svg+xml", headers=hdrs)
  ii = rclip_server.image_info_from_id(img_id)
  if ii.thumbnail_url:
    thm_url = ii.thumbnail_url
    thm_url = re.sub(r'/600px-',f'/{size}px-',thm_url)
    return fastapi.responses.RedirectResponse(thm_url)
  img = PIL.Image.open(ii.filename)

  
  EXIF_ORIENTATION = 0x0112
  ROTATION = {3: PIL.Image.ROTATE_180, 6: PIL.Image.ROTATE_270, 8: PIL.Image.ROTATE_90}
  code = img.getexif().get(EXIF_ORIENTATION, 1)
  #print(f"filename: {file};  EXIF rotation code: {code}")
  if code and code != 1:
    img = PIL.ImageOps.exif_transpose(img)

  img.thumbnail((size,size))
  buf = io.BytesIO()
  if img.mode != 'RGB': img = img.convert('RGB')
  img.save(buf,format="jpeg")
  buf.seek(0)
  return StreamingResponse(buf,media_type="image/jpeg", headers=hdrs)

@app.get("/info/{img_id}")
async def info(img_id:int):
    img_path = rclip_server.get_path_from_id(img_id)
    img     = PIL.Image.open(img_path)
    clip_embedding = rclip_server.get_image_embedding([img])
    info = {'path':img_path,'embedding':clip_embedding.tolist()}
    return fastapi.responses.JSONResponse(content=info)

@app.get("/copyright_message")
async def copyright_message():
    msg = {'copyright_message':rclip_server.copyright_message()}
    return msg

###############################################################################
# Launch the server
###############################################################################

if __name__ == "__main__":
  # uvicorn.run("rclip_server:app", host='0.0.0.0', port=8000, debug=True, reload=True)
  print("""
     Best run using a command like like:")
         uvicorn rclip_server:app")
     to launch an instance using the user's default rclip database, or")
         env CLIP_DB=wikimedia_quality.sqlite3 uvicorn rclip_server:app --reload")
     to run a dev instance specifying a different index")

      to debug:
          pyton
          >>> import rclip_server
          >>> rs = rclip_server.rclip_server
          >>> txt_embedding = rclip_server.rclip_server.get_text_embedding('search terms')
          >>> img_embeddings = rs.image_embeddings
          >>> similarities = (txt_embedding @ img_embeddings.T).squeeze(0).tolist()


    Note: If using nginx in front of uvicorn, add the following config to
          allow the large GET reqests with clip embedding paramaters:
          
              large_client_header_buffers 16 16k;
  """,file=sys.stderr)


