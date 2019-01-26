import pickle

data_items = []

with open('validation_data_items.pickle', 'rb') as f:
    while True:
        try:
            data_items.append(pickle.load(f))

        except EOFError:
            break

print(data_items)