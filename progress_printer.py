import numpy as np
import sys
import threading
import time
import datetime as dt


class ProgressPrinter(threading.Thread):
    def __init__(self, total):
        super().__init__()
        self.times = []
        self.processed = 0
        self.total = float(total)
        self.left = total
        self.mean = 1
        self.progress_str = '*[%s] %s%s %s. Time left: %s         \r'
        self.alive = True

    def print_progress(self, text='done'):
        bar_len = 60
        filled_len = int(round(bar_len * self.processed / self.total))
        percents = round(100.0 * self.processed / self.total, 1)
        bar = '█' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write(self.progress_str % (bar, percents, '%', text, dt.timedelta(seconds=self.get_eta())))
        sys.stdout.flush()

    def run(self):
        time.sleep(4)
        while self.alive:
            self.print_progress()
            time.sleep(1)

    def get_eta(self):
        return int(self.left * self.calculate_mean())

    def calculate_mean(self):
        if (self.processed < self.total * 0.1) \
                or (self.processed < 2000 and self.processed % 20 == 0) \
                or self.processed % 200 == 0:
            if self.times:
                self.mean = np.mean(self.times)
        return self.mean

    def register_progress_time(self, process_time):
        self.times.append(process_time)
        self.processed += 1
        self.left -= 1

    def kill(self):
        self.alive = False
