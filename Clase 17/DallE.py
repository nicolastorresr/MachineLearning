from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

import urllib.request
import base64
from IPython.display import *

def codificar_imagen(url):
    layout = ""   
    image = urllib.request.urlopen(url).read()
    image = base64.b64encode(image).decode('utf-8')
    img = f"""<img src="data:image/png;base64,{image}" height='60%' />"""
    return(img)

def codificar_imagenes(urls):
    layout = ""   
    for i in range(3):
        image = urllib.request.urlopen(urls[i]).read()
        image = base64.b64encode(image).decode('utf-8')
        img = f"""<div><img src="data:image/png;base64,{image}"/>"""
        layout += img[:-2] + ' align="left" width="30%"/></div>'
    return(layout)

def crear_imagen(prompt, model, n, size, quality):
    if model == "dall-e-3":
        n = 1
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
    )
    if model == "dall-e-3":
        image_url = response.data[0].url
        return codificar_imagen(image_url)
    if n > 3:
        n = 3
    image_urls = response.data[0].url,response.data[1].url,response.data[2].url 
    return codificar_imagenes(image_urls)

def mostrar_imagen(codigos):
    display(HTML(codigos))

def DALLE(prompt, model="dall-e-3", n=3, size="1024x1024", quality="standard"):
    imagenes = crear_imagen(prompt, model, n, size, quality)
    mostrar_imagen(imagenes)
