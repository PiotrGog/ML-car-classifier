from source.utils.history_help import *


history_name = "./history.hist"
h = load_history(history_name)

for l, a, lv, av in zip(h['loss'], h['accuracy'], h['val_loss'], h['val_accuracy']):
    print(f"{l:.4f}\t{a:.4f}\t{lv:.4f}\t{av:.4f}")
