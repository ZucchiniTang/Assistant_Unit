from time import sleep
from tqdm.auto import tqdm

for i in tqdm(range(10)):
    if i == 3:
        sleep(20)
        raise Exception("Wrong!")
    else:
        sleep(1)
