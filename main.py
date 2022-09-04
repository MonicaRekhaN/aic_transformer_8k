import numpy as np
import functions
import uvicorn
import cv2
from keras.models import Model
from starlette.responses import RedirectResponse
import tensorflow as tf

from fastapi.templating import Jinja2Templates
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
from main import *
import base64
from functions import Transformer, create_masks_decoder, evaluate, load_image
from pickle import load
from starlette.responses import Response
import uvicorn
from fastapi import FastAPI,Request,Form,UploadFile,File,BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import URL
from PIL import Image
from io import BytesIO
from keras.preprocessing import image
from keras.models import Model
from base64 import b64encode
import io
import numpy as np
from keras.preprocessing import image, sequence
from keras_preprocessing.sequence import pad_sequences
from keras.utils.image_utils import img_to_array,load_img  



app = FastAPI(Title="Image Captioning")
templates = Jinja2Templates(directory="templates")
templates.env.globals['URL'] = URL

# @app.get("/")
# async def root():

#     return {"message": "Hello World"}

@app.get("/",response_class=HTMLResponse)
def form_get(request: Request):
    print("testing")
#     global model
#     model=loadModel()
    return templates.TemplateResponse('form.html', context={'request': request})

@app.post("/after",response_class=HTMLResponse)
async def form_post(request: Request,file: UploadFile = File(...)):
    global model, resnet, vocab, inv_vocab
    print('start')
    file_contents = file.file.read()
    file_location = "file.jpg"
    with open(file_location, "wb+") as file_object:
        file_object.write(file_contents)

    f = file.file.read()
    npimg = np.fromstring(file_contents,np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.getvalue()).decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s"%(mime, img_base64)
    # f = file.file.read()
    # buf = BytesIO(f)
    # print('testing1')
    # contents = file.file.read()
    # base64_encoded_image = base64.b64encode(file_contents).decode("utf-8")

    caption = evaluate(file_location)

    #remove "<unk>" in result
    for i in caption:
        if i=="<unk>":
            caption.remove(i)

    #remove <end> from result         
    result_join = ' '.join(caption)
    result_final = result_join.rsplit(' ', 1)[0]
    print(result_final)
    print(result_join)
    return templates.TemplateResponse('after.html', context={'request': request,'data': result_join,'img_data': uri})
    
if __name__ == "__main__":
    functions.initialize()
    uvicorn.run(app, host="127.0.0.1", port=8000)
<<<<<<< HEAD


=======
>>>>>>> 005ce66b2bfff6c0d039dc4e436044dc29989db4
