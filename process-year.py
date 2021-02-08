import os
import threading

year = input("Enter the year: ")
processes = int(input("How many files to be processed at once?: "))

files = [year + "-%.2d" % i for i in range(1, 13)]


def run(filename):
    os.system(f"categorise.py {filename}")
    print(f"{filename} Finished")


for count, file in enumerate(files):
    t = threading.Thread(target=run, args=(file, ))
    t.start()
