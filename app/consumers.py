from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync
import json
from . import tasks
import logging
logging.basicConfig(filename="process.log", level=logging.INFO)

class ProcessConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        try:
            text_data_json = json.loads(text_data)
            
            async_to_sync(self.channel_layer.send)(self.channel_name, {"type": text_data_json['query'], "data": json.dumps(text_data_json)})
        except Exception as e:
            self.send(text_data=json.dumps({
                'message':'Failed'
            }))
            logging.exception(str(e))

    def predict(self, text_data):
        try:
            text_data_json = json.loads(text_data["data"])

            if not 'text' in text_data_json:
                return

            if not text_data_json['text']:
                return

            self.send(text_data=json.dumps({
                'message': 'Initializing process'
            }))
            tasks.predict.delay(self.channel_name, text_data_json)
        except Exception as e:
            logging.exception(str(e))

    def notify(self, text_data):
        try:
            self.send(text_data=json.dumps({
                'message': text_data["message"]
            }))
        except:
            self.send(text_data=json.dumps({
                'message': 'Failed'
            }))