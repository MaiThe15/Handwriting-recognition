import numpy as np
import gradio as gr
import cv2
 
import warnings
warnings.filterwarnings("ignore")
 
from tensorflow.keras.models import load_model
 
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
model = load_model('modelHandWritten.h5')
 
def classify(img):
    img = 255 - img
    img_final = cv2.resize(img, (28, 28))
    img_final = np.reshape(img_final, (1, 28, 28, 1))
    prediction = model.predict(img_final).flatten()
    return {word_dict[i]: float(prediction[i]) for i in range(25)}
 
iface = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="numpy", image_mode='L', sources=["upload", "clipboard"]),
    outputs=gr.Label(num_top_classes=3),
    live=True
)
 
 
if __name__ == "__main__":
    iface.launch(share=True)