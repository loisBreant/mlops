from kafka import KafkaProducer
import json
producer = KafkaProducer(bootstrap_servers=['nowledgeable.com:9092'])
producer.send('exo1', json.dumps({
    'data': [[1, 2,], [3, 78]]
    }).encode('utf-8'))
producer.flush()
producer.close()