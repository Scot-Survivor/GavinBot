"""
This is used to bulk run commands on my Databases w/ the reddit data
"""
import glob
import os
import sqlite3


timeframes = glob.glob("D:/Datasets/reddit_data/databases/*.db")
timeframes = [os.path.basename(timeframe) for timeframe in timeframes]

sql = """DELETE FROM parent_reply WHERE comment LIKE "%I don't know%" or comment like "%i don't know%";"""
for timeframe in timeframes:
    connection = sqlite3.connect('D:/Datasets/reddit_data/databases/{}'.format(timeframe))
    cursor = connection.cursor()
    cursor.execute(sql)
    connection.commit()
    print(f"{timeframe} SQL completed successfully")
