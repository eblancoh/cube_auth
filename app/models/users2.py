#!/usr/bin/env python

from sqlalchemy import (Column, String, Boolean, Integer, DateTime)
from sqlalchemy.orm import relationship

from . import db


# Declare the models of the DataBase
class Users2(db.Model):
    __tablename__ = 'Users2'

    id = Column(Integer, primary_key=True)
    user = Column(String, unique=True)
    session_token = Column(String)
    session_token_exp = Column(DateTime)
    solutions = Column(Integer, default=0)
    is_random = Column(Boolean)

    child = relationship("Movements2", back_populates="parent")

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def __init__(self, user):
        self.user = user
