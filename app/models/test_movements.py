#!/usr/bin/env python

from sqlalchemy import (Column, String, Boolean, Integer, Float)

from . import db


class TestMovements(db.Model):
    __tablename__ = 'test_movements'

    id = Column(Integer, primary_key=True)
    user = Column(String)
    user_id = Column(Integer)
    solution = Column(Integer)
    index = Column(Integer)
    stamp = Column(String)
    src = Column(String)
    event = Column(String)
    data_code = Column(String)
    data_order = Column(Integer)
    data_sync = Column(Boolean)
    data_lost = Column(Integer)
    data_side = Column(Integer)
    data_shift = Column(String)
    data_dbm = Column(Float)
    data_rssi = Column(Float)
    data_x = Column(Float)
    data_y = Column(Float)
    data_z = Column(Float)
    is_random = Column(Boolean)
    cube_type = Column(String)

    def __init__(self,
                 user, user_id, solution, index, stamp, src, event, data_code,
                 data_order, data_sync, data_lost, data_side, data_shift,
                 data_dbm, data_rssi, data_x, data_y, data_z, is_random,
                 cube_type):
        self.user = user
        self.user_id = user_id
        self.solution = solution
        self.index = index
        self.stamp = stamp
        self.src = src
        self.event = event
        self.data_code = data_code
        self.data_order = data_order
        self.data_sync = data_sync
        self.data_lost = data_lost
        self.data_side = data_side
        self.data_shift = data_shift
        self.data_dbm = data_dbm
        self.data_rssi = data_rssi
        self.data_x = data_x
        self.data_y = data_y
        self.data_z = data_z
        self.is_random = is_random
        self.cube_type = cube_type

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()
