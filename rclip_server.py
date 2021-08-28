#!/usr/bin/env python

import uvicorn
import html

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse
import io
import requests.utils
import subprocess
import PIL as pillow
from PIL import Image

app = FastAPI()

# https://stackoverflow.com/questions/55873174/how-do-i-return-an-image-in-fastapi
@app.get("/img")
async def main(q:str):
    path = q
    return FileResponse(path)

@app.get("/thm")
async def main(q:str):
    img = Image.open(q)
    thm = img.thumbnail((400,400))
    buf = io.BytesIO()
    img.save(buf,format="jpeg")
    buf.seek(0)
    return StreamingResponse(buf,media_type="image/jpeg")

@app.get("/search", response_class=HTMLResponse)
async def search(q:str):
    path = "/home/ron/tmp/old_pics_indexed_with_rclip/pics"
    cmd = f"(cd {path}; ~/proj/rclip/bin/rclip.sh -n --top 100 -f '{q}')"
    resp =  subprocess.run(cmd,shell=True,text=True,capture_output=True)
    result = resp.stdout
    results = result.splitlines()
    imgs = [f"""<a href="/img?q={i}" target="_blank"><img src="/thm?q={i}" style="max-width:400px;max-height:400px"></a>""" for i in results]
    return f"""<html>
       <div style="background-color: #ccc; width: 100%; padding: 30px; font-size: 20pt; font-family:Ariel">
       <form action="search">Search: <input name="q" value="{html.escape(q)}" style="width:800px"></form>
       </div>
       {" ".join(imgs)}
    </html>"""

@app.get("/",response_class=HTMLResponse)
def home():
    return """<html>
     <form action="search">Search: <input name="q" style="width:800px"></form>
    </html>"""

if __name__ == "__main__":
    uvicorn.run("rclip_server:app", host='0.0.0.0', port=8000, debug=True, reload=True)


# ~/proj/rclip/bin/rclip.sh -n dog -f | feh -f - t -g 1000x1000
