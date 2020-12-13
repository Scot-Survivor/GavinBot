import sqlite3
import pandas as pd
import multiprocessing as mp

timeframes = ['2015-01', '2014-12', '2014-11', '2014-10', '2014-09', '2014-01', '2014-02', '2014-05']


def sort_out(time_frame):
    connection = sqlite3.connect('D:/Datasets/reddit_data/databases/{}.db'.format(time_frame))
    limit = 1000000
    last_unix = 0
    cur_length = limit
    counter = 0

    while cur_length == limit:

        df = pd.read_sql(
            "SELECT * FROM parent_reply WHERE unix > {} and parent NOT NULL and score > 0 ORDER BY unix ASC LIMIT {}".format(
                last_unix, limit), connection)
        last_unix = df.tail(1)['unix'].values[0]
        cur_length = len(df)
        with open('train.from', 'a', encoding='utf8') as f:
            for content in df['parent'].values:
                f.write(content + '\n')

        with open('train.to', 'a', encoding='utf8') as f:
            for content in df['comment'].values:
                f.write(str(content) + '\n')

        counter += 1
        if counter % 2 == 0:
            print(counter * limit, ' rows completed so far' + f' {time_frame}')


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    cpus = 8
    chunked_list = chunks(timeframes, cpus)
    for cpu in range(cpus):
        p = mp.Process(target=sort_out, args=(next(chunked_list),), name=f"CPU-Sort Out")
