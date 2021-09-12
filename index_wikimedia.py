#!/usr/bin/env python
'''https://meta.wikimedia.org/wiki/User-Agent_policy

    https://commons.wikimedia.org/wiki/Commons:Valued_images

    There are currently 38,837 images marked as valued images, which
    is roughly 0.051% of the available images (76,678,602), and 283
    valued image sets. See also;

    https://commons.wikimedia.org/wiki/Commons:Featured_pictures

    There are currently 14,892 of such images in the Commons
    repository which is roughly 0.019% of the available images
    (76,673,875).

    https://commons.wikimedia.org/wiki/Commons:Quality_images

    There are currently 276,020 images marked as Quality images, which
    is roughly 0.36% of the available images (76,670,770).

    https://commons.wikimedia.org/wiki/Category:Featured_pictures_on_Wikimedia_Commons
    https://commons.wikimedia.org/wiki/Category:Quality_images
    https://commons.wikimedia.org/wiki/Category:Valued_images


    Note that it gives the occasional:

    b'<!DOCTYPE html>\n<html lang="en">\n<meta charset="utf-8">\n<title>Wikimedia Error</title>\n<style>\n* { margin: 0; padding: 0; }\nbody { background: #fff; font: 15px/1.6 sans-serif; color: #333; }\n.content { margin: 7% auto 0; padding: 2em 1em 1em; max-width: 640px; }\n.footer { clear: both; margin-top: 14%; border-top: 1px solid #e5e5e5; background: #f9f9f9; padding: 2em 0; font-size: 0.8em; text-align: center; }\nimg { float: left; margin: 0 2em 2em 0; }\na img { border: 0; }\nh1 { margin-top: 1em; font-size: 1.2em; }\n.content-text { overflow: hidden; overflow-wrap: break-word; word-wrap: break-word; -webkit-hyphens: auto; -moz-hyphens: auto; -ms-hyphens: auto; hyphens: auto; }\np { margin: 0.7em 0 1em 0; }\na { color: #0645ad; text-decoration: none; }\na:hover { text-decoration: underline; }\ncode { font-family: sans-serif; }\n.text-muted { color: #777; }\n</style>\n<div class="content" role="main">\n<a href="https://www.wikimedia.org"><img src="https://www.wikimedia.org/static/images/wmf-logo.png" srcset="https://www.wikimedia.org/static/images/wmf-logo-2x.png 2x" alt="Wikimedia" width="135" height="101">\n</a>\n<h1>Error</h1>\n<div class="content-text">\n<p>Our servers are currently under maintenance or experiencing a technical problem.\n\nPlease <a href="" title="Reload this page" onclick="window.location.reload(false); return false">try again</a> in a few&nbsp;minutes.</p>\n\n<p>See the error message at the bottom of this page for more&nbsp;information.</p>\n</div>\n</div>\n<div class="footer"><p>If you report this error to the Wikimedia System Administrators, please include the details below.</p><p class="text-muted"><code>Request from 2607:fb90:30dd:6246:3e82:19c7:c09:a89f via cp4022 cp4022, Varnish XID 756016548<br>Upstream caches: cp4022 int<br>Error: 429, Too Many Requests at Tue, 07 Sep 2021 21:33:56 GMT</code></p>\n</div>\n</html>\n'


'''

import PIL as pillow
import argparse
import clip
import clip.model
import datetime
import io
import mwclient
import numpy as np
import re
import requests
import sqlite3
import sys
import torch
import torch.nn
import filelock
lock = filelock.FileLock('/tmp/index_wikipedia.lock',timeout=10)

# get db
parser                    = argparse.ArgumentParser()
default_db                = './wikimedia_images.sqlite3'
parser.add_argument('--db','-d',default=str(default_db),help="path to rclip's database")
args,unk                  = parser.parse_known_args()

def create_table(dbname):
    sql = '''
        CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        deleted BOOLEAN,
        filepath TEXT NOT NULL UNIQUE,
        modified_at DATETIME,
        size INTEGER,
        vector BLOB NOT NULL,
        wikimedia_descr_url TEXT,
        wikimedia_thumb_url TEXT
      )
    '''
    with sqlite3.connect(dbname) as con:
        con.execute(sql)
        con.commit()

def get_already_processed_images(dbname):
    sql = '''select wikimedia_descr_url from images'''
    with sqlite3.connect(dbname) as con:
        con.row_factory = sqlite3.Row
        return set([row['wikimedia_descr_url'] for row in con.execute(sql)])
        
def check_pic(dbname,descr_url):
    sql = '''select wikimedia_descr_url from images where wikimedia_descr_url == :descr_url'''
    with sqlite3.connect(dbname) as con:
        con.row_factory = sqlite3.Row
        rows = list(con.execute(sql,{'descr_url':descr_url}))
        return rows

def save_image(dbname,descr_url:str, thm_url:str, clip_embedding:np.ndarray,size):
    clip_embedding_asbytes = clip_embedding.tobytes()
    sql = '''
      INSERT INTO images(filepath, wikimedia_descr_url, wikimedia_thumb_url, vector, size)
      VALUES (:filepath, :wikimedia_descr_url, :wikimedia_thumb_url, :vector, :size)
      ON CONFLICT(filepath) DO UPDATE SET
        vector=:vector
    '''
    with lock:
        with sqlite3.connect(dbname) as con:
            con.execute(sql, {
                       'filepath'            : descr_url,
                       'wikimedia_descr_url' : descr_url,
                       'wikimedia_thumb_url' : thm_url,
                       'vector'              : clip_embedding_asbytes,
                       'size'                : size
                       })
            con.commit()

def get_images_in_category(category_name):
    site = mwclient.Site('commons.wikimedia.org')
    category = site.Categories[category_name]
    images = (x for x in category.members() if isinstance(x,mwclient.image.Image))
    return images

clip_model, clip_preprocess = clip.load('ViT-B/32','cpu')

# https://meta.wikimedia.org/wiki/User-Agent_policy
wikimedia_api_headers = {'User-agent': 
                         "OpenAI Clip Embedding Calculator/0.0 (https://github.com/ramayer/wikipedia_in_spark; ramayer+git@gmail.com) generic-library/0.0"}

def process_image(image_url,descr_url):
    if (image_url.endswith('.svg') or 
        image_url.endswith('.webm') or
        image_url.endswith('.stl') or
        image_url.endswith('.tif') or
        image_url.endswith('.tiff') or
        image_url.endswith('.ogv')
        ):
        print(f"can't handle non-pillow.Image image types like {image_url} yet")
        return(None,None,None,None)
    ext = re.sub(r'.*\.','',image_url)
    if ext.lower() not in {'jpg', 'jpeg', 'JPG', 'PNG', 'JPEG', 'png', 'gif'}:
        print(f"not sure if it can handle image types like {image_url} yet")
        return(None,None,None,None)
    print(' ',datetime.datetime.now().isoformat(),image_url,descr_url)
    sys.stdout.flush()
    url_suffix   = re.sub(r'.*/','',image_url)
    thm_url      = re.sub('/commons/','/commons/thumb/',image_url) + '/600px-' + url_suffix
    headers      = wikimedia_api_headers
    response     = requests.get(thm_url,headers=headers)
    content      = response.content
    try:
        pil_img      = pillow.Image.open(io.BytesIO(content))
    except Exception as e:
        print(e)
        print(content)
        raise e

    with torch.no_grad():
        preprocessed = clip_preprocess(pil_img)
        encoded      = clip_model.encode_image(torch.stack([preprocessed]))
        norm         = encoded.norm(dim=-1,keepdim=True)
        normed       = encoded / norm
    clip_embedding = normed.cpu().numpy()
    return(descr_url,thm_url,clip_embedding,len(content))

create_table(args.db)

already_done = set(get_already_processed_images('wikimedia_images.sqlite3'))

sane_cat = 'Valued_images_promoted_2021-08'
big_cat = 'Featured_pictures_on_Wikimedia_Commons'
bigger_cat = 'Valued_images'
huge_cat = 'Quality_images'
cat = 'Kung_fu'
cat = None

if cat:
    for img in get_images_in_category(cat):
        descr_url    = img.imageinfo['descriptionurl']
        if descr_url in already_done:
            print(f"already done: {descr_url}")
        else:
            title        = img.base_title
            name         = img.name
            image_url    = img.imageinfo['url']
            descr_url    = img.imageinfo['descriptionurl']

            descr_url,thm_url,clip_embedding,length = process_image(image_url,descr_url)
            print(' ',descr_url,length)
            if clip_embedding is not None:
                save_image(args.db,descr_url,thm_url,clip_embedding,length)

import json
file = 'quality_metadata.ndjson'
import random
with open(file) as f:
    lines = [json.loads(l) for l in f.readlines()]
random.shuffle(lines)

for idx,l in enumerate(lines):
    image_url = l['url']
    descr_url = l['descriptionurl']
    if descr_url in already_done:
        print(f"already done: {descr_url}")
        continue
    if check_pic(args.db,descr_url):
        print(f"recently done: {descr_url}")
        continue

    descr_url,thm_url,clip_embedding,length = process_image(image_url,descr_url)
    print('indexing ',descr_url,length)
    if clip_embedding is not None:
        save_image(args.db,descr_url,thm_url,clip_embedding,length)

