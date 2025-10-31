from kafka import KafkaProducer
producer = KafkaProducer(bootstrap_servers=['nowledgeable.com:9092'])
producer.send('exo1', b'coucou lois.breant')
producer.flush()
producer.close()