_device = None


def get_device():
    return _device


def set_device(dev):
    global _device
    _device = dev
