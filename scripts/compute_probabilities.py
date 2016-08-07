import scripts
import os

import stormpy
import stormpy.logic

BASE_DIR = 'data/learning-tables/learning-4-july-blade21/'
# BASE_DIR = 'data/repair'
temperature = 10

def main():
    file_name = 'dtmc-sm{}'.format(temperature)
    fallen_formula = stormpy.parse_formulas("P=? [ F \"fallen\" ]")
    goal_formula = stormpy.parse_formulas("P=? [ F \"goal\" ]")
    far_formula = stormpy.parse_formulas("P=? [ F \"far\" ]")
    collided_formula = stormpy.parse_formulas("P=? [ F \"collided\" ]")
    total_formula = stormpy.parse_formulas("P=? [ F (\"far\"  | \"collided\" | \"fallen\")]")

    model = stormpy.parse_explicit_model(os.path.join(BASE_DIR, file_name + '.tra'),
                                             os.path.join(BASE_DIR, file_name + '.lab'))
    fallen_prob = stormpy.model_checking(model, fallen_formula[0])
    goal_prob = stormpy.model_checking(model, goal_formula[0])
    far_prob = stormpy.model_checking(model, far_formula[0])
    collided_prob = stormpy.model_checking(model, collided_formula[0])
    total_prob = stormpy.model_checking(model, total_formula[0])
    print('Fallen Prob: {}'.format(fallen_prob))
    print('Goal Prob: {}'.format(goal_prob))
    print('Far Prob: {}'.format(far_prob))
    print('Collided Prob: {}'.format(collided_prob))
    print('Total: {}'.format(total_prob))

if __name__ == '__main__':
    main()
