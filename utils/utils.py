import sys


class StaticPrinter:
    def __init__(self):
        self.num_lines = 0

    def print(self, line):
        print(line)
        self.num_lines += 1

    def reset(self):
        for _ in range(self.num_lines):
            sys.stdout.write("\033[F")  # Cursor up one line
            sys.stdout.write("\033[K")  # Clear to the end of line
        self.num_lines = 0
