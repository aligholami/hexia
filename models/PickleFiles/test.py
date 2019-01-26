import pickle

data_items = []

with open('validation_data_items.txt', 'rb') as f:
    while True:
        try:
            data_items.append(pickle.load(f))

        except EOFError:
            break


for item in data_items:
    for img, sent, conf in item:
        print("item 1: {}, {}, {}".format(img, sent, conf))
