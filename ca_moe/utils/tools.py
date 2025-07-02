import logging

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