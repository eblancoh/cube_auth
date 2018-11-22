#!/usr/bin/env python

from sqlalchemy import (Column, String, Boolean, Integer, DateTime)
from sqlalchemy.orm import relationship

from . import db


# Declare the models of the DataBase
# Table Users in linked to Movements in a parent -> child relationship
class Users(db.Model):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    user = Column(String, unique=True)
    session_token = Column(String)
    session_token_exp = Column(DateTime)
    # session_token_usage = Column(Integer, default=0)
    # session_token_valid = Column(Boolean)
    solutions = Column(Integer, default=0)
    # auth_threshold = Column(Integer, default=auth_thres)
    is_random = Column(Boolean)
    child = relationship("Movements", back_populates="parent")

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def __init__(self, user):
        self.user = user
