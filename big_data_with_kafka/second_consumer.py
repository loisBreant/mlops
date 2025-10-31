from kafka import KafkaConsumer
import json
import numpy as np

consumer = KafkaConsumer('exo1', bootstrap_servers=['nowledgeable.com:9092'])
for message in consumer:
    try:
        data = json.loads(message.value.decode('utf-8'))
        array = np.array(dict(data)['data'])
        print('Type:', type(array))
        print('Array:', array)
    except json.JSONDecodeError:
        print("Message is not in JSON format:", message.value.decode('utf-8'))
        continue

# Les deux consumers peuvent coexister et lire les messages du même topic.
    