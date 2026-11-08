import importlib


class ConsoleLogger:
    def log(self, data: dict[str, float], step: int) -> None:
        metrics = " ".join(f"{key}={value:.6g}" for key, value in data.items())
        print(f"step {step}: {metrics}", flush=True)


class WandbLogger:
    def __init__(
        self, project: str, name: str | None, config: dict[str, object]
    ) -> None:
        # Load wandb lazily so it only have to be installed when used
        self.wandb = importlib.import_module("wandb")
        self.run = self.wandb.init(project=project, name=name, config=config)
        self.console = ConsoleLogger()

    def log(self, data: dict[str, float], step: int) -> None:
        self.run.log(data, step=step)
        self.console.log(data, step)

    def finish(self) -> None:
        self.run.finish()
