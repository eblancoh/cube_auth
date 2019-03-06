#!/usr/bin/env python

import pika

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()



channel.queue_declare(queue='testing queue')

channel.basic_publish(exchange='',
                      routing_key='testing queue',
                      body='{"email": "julio@example.com","movements": [{"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:58:57.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:58:56.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:56.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:00.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:30.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:40.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:45.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:55.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:30.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:40.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:45.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:55.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:58:57.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:58:56.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:56.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:00.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:30.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:40.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:45.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:55.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:30.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:40.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:45.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:55.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:58:57.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:58:56.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:56.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:00.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:30.302Z"}, {"email": "julio@example.com", "yaw_pitch_roll": [{"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}, {"x": 10, "y": 180, "z": 3}], "cube_type": "11paths", "turn": "41", "sequence": "string", "is_random": true, "timestamp": "2018-08-17T09:59:40.302Z"}],"id": "fsldkfjfjdsfjd"}')
print(" [x] Message Sent !")
connection.close()
