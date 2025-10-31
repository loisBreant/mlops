from kafka import KafkaProducer, KafkaConsumer
import json
import numpy as np

consumer = KafkaConsumer('exo1', bootstrap_servers=['nowledgeable.com:9092'])
producer = KafkaProducer(bootstrap_servers=['nowledgeable.com:9092'])

for message in consumer:
    try:
        data = json.loads(message.value.decode('utf-8'))
        array = np.array(dict(data)['data'])
        sum_array = np.sum(array)
        producer.send('processed',        
            json.dumps({'sum': sum_array}).encode('utf-8'))
        producer.flush()
        
    except json.JSONDecodeError:
        print("Message is not in JSON format:", message.value.decode('utf-8'))
        continue

producer.close()

# Expliquer ce qu'est un consumer en kafka 
# Expliquer ce qu'est un producer en kafka

# Un producer kafka est