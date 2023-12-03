from os.path import join

from src.hparams.hparams import HParams

class HParamsSac(HParams):
    def __init__(self, path = None) -> None:
        super().__init__()
        self.default_path = join("src", "configs", "hparams", "default_sac.yml")
        super().load(self.default_path) if path is None else super().load(path)
        

if __name__ == "__main__":
    hparams = HParamsSac()
    print(hparams)
