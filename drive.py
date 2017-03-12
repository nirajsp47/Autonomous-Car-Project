import theano
import keras
from keras.models import load_model
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from flask import Flask, render_template

sio = socketio.Server()
app = Flask(__name__)

@sio.on('telemetry')
def telemetry(sid, data):
    throttle = 0.2
    steering_angle=0
    send_control(steering_angle, throttle)
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    '''Write image pre processing function here and then pass that image as an array to the model to predict the steering value'''
    
     


    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
    print(speed,throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    

