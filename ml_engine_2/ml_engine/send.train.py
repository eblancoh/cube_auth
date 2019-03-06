#!/usr/bin/env python

import pika

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='training queue')

channel.basic_publish(exchange='',
                      routing_key='training queue',
                      body='{"email": "julio@example.com"}')
print(" [x] Message Sent !")
connection.close()
