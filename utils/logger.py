import os

class Logger:
    def __init__(self, log_file='outputs/metrics/log.txt'):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.file = open(log_file, 'w')

    def log(self, epoch, d_loss, g_loss):
        line = f"Epoch {epoch}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}\n"
        self.file.write(line)
        self.file.flush()

    def __del__(self):
        self.file.close()