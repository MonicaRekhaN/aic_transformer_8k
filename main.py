import numpy as np
import functions
import uvicorn
import cv2
from keras.models import Model
import tensorflow as tf
from keras.applications.resnet import ResNet50
from fastapi.templating import Jinja2Templates
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
from main import *
import base64
from functions import Transformer, create_masks_decoder, evaluate, load_image
from pickle import load
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

# def initialize():
num_layer = 4
d_model = 512
dff = 2048
num_heads = 8
row_size = 7
col_size = 7
top_k=5000
target_vocab_size = top_k + 1 # top_k = 5000
dropout_rate = 0.1
global tokenizer,image_features_extract_model,transformer,output,dec_mask
global start_token,decoder_input,end_token
#Building a Word embedding for top 5000 words in the captions
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                oov_token="<unk>",
                                                filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
pkl_tokenizer_file="encoded_tokenizer.pkl"
# Load the tokenizer train features to disk
with open(pkl_tokenizer_file, "rb") as encoded_pickle:
    tokenizer = load(encoded_pickle)
#Image Model
image_model = ResNet50(include_top=False,weights='imagenet',input_shape=(224, 224,3),pooling="avg")
new_input = image_model.input
hidden_layer = image_model.layers[-2].output
print("in init")
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
transformer = Transformer(num_layer,d_model,num_heads,dff,row_size,
                col_size,target_vocab_size,max_pos_encoding=target_vocab_size,rate=dropout_rate)
# transformer()
start_token = tokenizer.word_index['<start>']
end_token = tokenizer.word_index['<end>']
decoder_input = [start_token]
output = tf.expand_dims(decoder_input, 0) #token
dec_mask = create_masks_decoder(output)
test = tf.random.Generator.from_seed(123)
test = test.normal(shape=(16,49,2048))
transformer(test,output,False,dec_mask)
transformer.load_weights('model.h5')


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

@app.post("/predict",response_class=HTMLResponse)
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
    base64_encoded_image = base64.b64encode(file_contents).decode("utf-8")

    global tokenizer,image_features_extract_model,transformer,output,dec_mask
    global start_token,decoder_input,end_token
    temp_input = tf.expand_dims(load_image(file_location)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    result = [] #word list
  
    for i in range(100):
      dec_mask = create_masks_decoder(output)
      predictions = transformer(img_tensor_val,output,False,dec_mask)
      # select the last word from the seq_len dimension
      predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
      predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
      # return the result if the predicted_id is equal to the end token
      if predicted_id == end_token:
          break #return result
      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      result.append(tokenizer.index_word[int(predicted_id)])
      output = tf.concat([output, predicted_id], axis=-1)

    #remove "<unk>" in result
    for i in result:
        if i=="<unk>":
            result.remove(i)

    #remove <end> from result         
    result = ' '.join(result)
    print(result)
    return templates.TemplateResponse('predict.html', context={'request': request,'data': result,'img_data': uri})
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
