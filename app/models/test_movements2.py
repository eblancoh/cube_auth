#!/usr/bin/env python

from sqlalchemy import (Column, String, Boolean, Integer, Float)

from . import db


class TestMovements2(db.Model):
    __tablename__ = 'TestMovements2'

    id = Column(Integer, primary_key=True)
    user = Column(String)
    user_id = Column(Integer)
    solution = Column(Integer)
    stamp = Column(String)
    turn = Column(String)
    is_random = Column(Boolean)
    cube_type = Column(String)
    frequency = Column(Integer)
    YawPitchRoll = Column(String)

    def __init__(self,
                user, user_id, solution, stamp, is_random, 
                turn, frequency, YawPitchRoll, cube_type):
        self.user = user
        self.user_id = user_id
        self.solution = solution
        self.stamp = stamp
        self.is_random = is_random
        self.turn = turn
        self.frequency = frequency
        self.cube_type = cube_type
        self.YawPitchRoll = YawPitchRoll

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()
