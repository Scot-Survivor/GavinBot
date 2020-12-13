import sqlite3
import os

if not os.path.exists("chatLogs.db"):
    connection = sqlite3.connect("chatLogs.db")
    c = connection.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS chat_logs(guild TEXT, channel TEXT, model TEXT, author TEXT, ctx TEXT, reply TEXT, date TEXT)")
    connection.commit()
    del c
    del connection


def connect() -> sqlite3.Connection and sqlite3.Cursor:
    returnConnection = sqlite3.connect("chatLogs.db")
    returnCursor = returnConnection.cursor()
    return returnConnection, returnCursor


async def sql_insert_into(guild: str, channel: str, model: str, author: str, message: str, reply: str, date: str, u_connection: sqlite3.Connection, cursor: sqlite3.Cursor):
    sql = f"""INSERT INTO chat_logs (guild, channel, model, author, ctx, reply, date) VALUES ('{guild}', '{channel}', '{model}', '{author}', '{message}', '{reply}', '{date}')"""
    try:
        cursor.execute(sql)
        u_connection.commit()
    except Exception as e:
        return f"Insertion Error: {e}"
    else:
        return True
