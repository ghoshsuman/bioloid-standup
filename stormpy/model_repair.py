from stormpy import stormpy
from stormpy.stormpy import logic


class ModelRepairer:

    def __init__(self):
        self.formula = stormpy.parse_formulas("P=? [ F \"fallen\" ]")

    def repair(self, ):
        model = stormpy.parse_explicit_model("die.tra", "die.lab")


    def local_repair(self):
        pass



def main():
    model_repairer = ModelRepairer()


if __name__ == '__main__':
    main()