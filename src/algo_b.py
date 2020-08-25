import Algorithmia
import tensorflow as tf
import logging
import shutil
import os
from time import time
client = Algorithmia.client()

SM_PATH = "/tmp/saved_model"
MODEL_PATH = "data://network_anomaly_detection/models/modelb_tf_23_aug2520.zip"
LOGGING_FILE = "/tmp/logging_file"

class TF_Logging():
    def __init__(self, logging_file):
        self.log_file = logging_file
        self.get_logger()
        self.events = []

    def insert_event(self, message):
        event = {"timestamp": str(time()), "message": message}
        self.events.append(event)

    def parse_logs(self):
        with open(self.log_file, 'r') as f:
            for line in f.readline():
                elements = line.split(' - ')
                timestamp = str(elements[0])
                message = str(elements[-1])
                event = {"timestamp": timestamp, "message": message}
                self.events.append(event)
        os.remove(self.log_file)
        open(self.log_file, 'w').close()

    def get_events(self):
        events = self.events
        self.events = []
        return events

    def get_logger(self):
        log = logging.getLogger("tensorflow")
        log.setLevel("DEBUG")
        formatter = logging.Formatter('%(created)f - %(levelname)s - %(message)s')
        fh = logging.FileHandler(self.log_file)
        fh.setFormatter(formatter)
        log.addHandler(fh)



def load():
    local_path = client.file(MODEL_PATH).getFile().name
    shutil.unpack_archive(local_path, SM_PATH, format='zip')
    model = tf.keras.models.load_model(SM_PATH, compile=False)
    return model


def apply(input):
    LOGGER.insert_event("input to ALGO_B is: {}".format(input))
    tensor = tf.constant(input)
    outcome = MODEL.predict(tensor).tolist()[0][0]
    LOGGER.parse_logs()
    LOGGER.insert_event("outcome predicted by ALGO_B is: {}".format(outcome))
    events = LOGGER.get_events()
    output = {"outcome": outcome, "events": events}
    return output


LOGGER = TF_Logging(LOGGING_FILE)
MODEL = load()

if __name__ == "__main__":
    output = apply([[4.0, 3.0, 2.2, -1.1, 0.25]])
    print(output)
    output = apply([[4.0, 2.9, 2.2, -1.1, 0.25]])
    print(output)
    output = apply([[4.0, 3.1, 2.2, -1.1, 0.25]])
    print(output)
