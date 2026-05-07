from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.sparsity import LASSOSparsity
from epde.operators.utils.operator_mappers import map_operator_between_levels
from epde.operators.utils.template import CompoundOperator
import epde.operators.common.fitness as fitness

class FitnessOperatorFactory:
    @staticmethod
    def create(name: str, params: dict) -> CompoundOperator:
        cls_map = {
            "PIC": fitness.PIC,
            "DeepXDEBasedFitness": fitness.DeepXDEBasedFitness,
            "L2LRFitness": fitness.L2LRFitness,
        }
        if name not in cls_map:
            raise ValueError(f"Unknown operator: {name}")

        operator = cls_map[name](list(params.keys()))
        sparsity = LASSOSparsity()
        coeff_calc = LinRegBasedCoeffsEquation()
        if name == 'PIC':
            sparsity = map_operator_between_levels(sparsity, 'gene level', 'chromosome level')
            coeff_calc = map_operator_between_levels(coeff_calc, 'gene level', 'chromosome level')

        operator.set_suboperators({
            "sparsity": sparsity,
            "coeff_calc": coeff_calc,
        })
        operator.params = params

        if 'chromosome level' not in operator._tags:
            fitness_cond = lambda x: not getattr(x, "fitness_calculated", False)
            operator = map_operator_between_levels(
                operator,
                'gene level',
                "chromosome level",
                objective_condition=fitness_cond,
            )
        return operator