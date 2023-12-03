import yaml

class HParams():
    def load(self, path):
        with open(path, 'r') as infile:
            update_dict = yaml.safe_load(infile)
        self.update(update_dict)
        return self

    def save(self, path):
        print(vars(self))
        defined_variables = {key: value for key, value in vars(self).items() if "__" not in key}
        with open(path, 'w') as outfile:
            yaml.dump(defined_variables, outfile)
        return self

    def update(self, update_dict):
        for key, value in update_dict.items():
            setattr(self, key, value)

    def __repr__(self):
        return 'Hyperparameters Object:\n'+''.join([f"{key}: {value}\n" for key, value in vars(self).items()])