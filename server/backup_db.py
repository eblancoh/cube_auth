#!/usr/bin/env python

"""
This script creates a timestamped database backup,
and cleans backups older than a set number of dates

"""

from __future__ import print_function
from __future__ import unicode_literals

import configparser
import os
import shutil
import sqlite3
import time

description = """
              Create a timestamped SQLite database backup, and
              clean backups older than a defined number of days
              """

basedir = os.path.abspath(os.path.dirname(__file__))

# How old a file needs to be in order to be considered for being removed
# Cargamos la configuración
cfg = configparser.ConfigParser()
cfg.read(os.path.join(basedir, '..', 'config', 'cfg.ini'))
no_of_days = cfg.getint('database', 'NO_OF_DAYS')


def sqlite3_backup(dbfile, backupdir):
    """Create timestamped database copy"""

    if not os.path.isdir(backupdir):
        os.mkdir(backupdir)
        # raise Exception("Backup directory does not exist: {}".format(backupdir))

    backup_file = os.path.join(backupdir, os.path.basename(dbfile) +
                               time.strftime("-%Y%m%dT%H%M%S"))

    connection = sqlite3.connect(dbfile)
    cursor = connection.cursor()

    # Lock database before making a backup
    cursor.execute('begin immediate')
    # Make new backup file
    shutil.copyfile(dbfile, backup_file)
    print("\nCreating {}...".format(backup_file))
    # Unlock database
    connection.rollback()


def clean_data(backup_dir):
    """Delete files older than NO_OF_DAYS days"""

    print("\n------------------------------")
    print("Cleaning up old backups")

    for filename in os.listdir(backup_dir):
        backup_file = os.path.join(backup_dir, filename)
        if os.stat(backup_file).st_ctime < (time.time() - no_of_days * 86400):
            if os.path.isfile(backup_file):
                os.remove(backup_file)
                print("Deleting {}...".format(backup_file))
