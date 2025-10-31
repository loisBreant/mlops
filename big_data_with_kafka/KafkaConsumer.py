from kafka import KafkaConsumer

consumer = KafkaConsumer('exo1', bootstrap_servers=['nowledgeable.com:9092'])
for message in consumer:
    print(message.value.decode('utf-8'))