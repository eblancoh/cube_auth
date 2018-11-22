#!/usr/bin/env python

from sqlalchemy import (Column, String, Integer)

from . import db


# Create revoked token model:
class RevokedTokenModel(db.Model):
    __tablename__ = 'revoked_tokens'
    id = Column(Integer, primary_key=True)
    jti = Column(String(120))

    def add(self):
        db.session.add(self)
        db.session.commit()

    @classmethod
    def is_jti_blacklisted(cls, jti):
        query = cls.query.filter_by(jti=jti).first()
        return bool(query)
