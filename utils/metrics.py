class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc..
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg


class AverageMeterList:
    """
    Class to be an average meter for any average metric List structure like mean_iou_per_class
    """

    def __init__(self, num_cls):
        self.num_cls = num_cls
        self.value = [0] * self.num_cls
        self.avg = [0] * self.num_cls
        self.sum = [0] * self.num_cls
        self.count = [0] * self.num_cls
        self.reset()

    def reset(self):
        self.value = [0] * self.num_cls
        self.avg = [0] * self.num_cls
        self.sum = [0] * self.num_cls
        self.count = [0] * self.num_cls

    def update(self, val, n=1):
        for i in range(self.num_cls):
            self.value[i] = val[i]
            self.sum[i] += val[i] * n
            self.count[i] += n
            self.avg[i] = self.sum[i] / self.count[i]

    @property
    def val(self):
        return self.avg


class EarlyStopping:
    def __init__(self, mode, patience, verbose=False):
        self.step_i = 0
        self.patience = patience
        self.mode = mode
        self.verbose = verbose

        if self.mode == "max":
            self.best_score = -float("inf")
        elif self.mode == "min":
            self.best_score = float("inf")
        else:
            raise ValueError(f"Unknown {mode}")

    def step(self, new_score):
        if self.mode == "max":
            is_best = new_score > self.best_score
        else:
            is_best = new_score < self.best_score

        if is_best:
            self.best_score = new_score
            self.step_i = 0
        else:
            self.step_i += 1

        if self.verbose:
            print(f"EarlyStopping {self.step_i}/{self.patience}")

        if self.step_i > self.patience:
            return True
        else:
            return False
