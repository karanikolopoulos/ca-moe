import logging

from omegaconf import ListConfig


# https://github.com/facebookresearch/hydra/blob/main/plugins/hydra_colorlog/hydra_plugins/hydra_colorlog/conf/hydra/hydra_logging/colorlog.yaml
class CustomFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if "ca_moe" in record.pathname:
            module = "CA-MoE"

        return "[{} - {}] [{} {}:{}] {}".format(
            module,
            record.levelname,
            self.formatTime(record, datefmt="%m-%d %H:%M:%S"),
            record.filename,
            record.lineno,
            record.getMessage(),
        )


def flatten(shape: list[int]) -> int:
    num_input_fmaps = 1
    for s in shape:
        num_input_fmaps *= s
    return num_input_fmaps


def calc_channels(*args):
    (args,) = args
    height, *_ = args.get("input_shape")
    filters = args.get("filters")
    padding = args.get("padding")
    kernel = args.get("kernel_size")
    strides = args.get("strides")

    if padding == "valid":
        pad = 0
    elif padding == "same":
        pad = (kernel - 1) / 2
    else:
        raise TypeError

    channel = int(((height + 2 * pad - kernel) / strides) + 1)

    return ListConfig([channel, channel, filters])
