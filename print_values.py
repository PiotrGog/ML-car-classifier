from source.utils.history_help import *


history_name = "./resnet_pg224_128_20_0.0001.hist"
h = load_history(history_name)
print(h['accuracy'])
print(h.keys())

for l, a, lv, av in zip(h['loss'], h['accuracy'], h['val_loss'], h['val_accuracy']):
    print(f"{l:.4f}\t{a:.4f}\t{lv:.4f}\t{av:.4f}")
