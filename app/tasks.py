import json
import logging
logging.basicConfig(filename="process.log", level=logging.INFO)

from FinalProject.celery import app
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from app.machine_learning.classifier import Classifier

channel_layer = get_channel_layer()

@app.task
def predict(channel_name, data):
    try:
        model = None
        if data['app'].lower() == 'newsreliability':
            model = 'maxent'
        elif data['app'].lower() == 'predictingtruthfullness':
            if data['model']:
                model = data['model']
            
        async_to_sync(channel_layer.send)(channel_name, {"type": "notify", "message": "Predicting text"})
        
        classifier = Classifier(data['app'], model)
        result = classifier.predict([data['text']])

        if result:
            async_to_sync(channel_layer.send)(channel_name, {"type": "notify", "message": str(result)})
        else:
            async_to_sync(channel_layer.send)(channel_name, {"type": "notify", "message": "Failed"})
    except Exception as e:
        logging.exception(str(e))