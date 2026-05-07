from abc import ABC, abstractmethod
import numpy as np
from epde.interface.equation_translator import translate_equation
from epde.interface.token_family import TFPool
from epde.structure.main_structures import SoEq, Chromosome
import copy


class ComparisonStrategy(ABC):
    @abstractmethod
    def build(self, symbolic, search_obj, all_vars):
        pass

    @abstractmethod
    def compare(self, correct_obj, incorrect_obj, fit_operator, all_vars):
        pass


class SingleEquationComparison(ComparisonStrategy):
    def build(self, symbolic, search_obj, all_vars):
        metaparams = {("sparsity", var): {"optimizable": False, "value": 1e-6} for var in all_vars}
        eq = translate_equation(symbolic, search_obj.pool, all_vars=all_vars)
        for var in all_vars:
            eq.vals[var].main_var_to_explain = var
            eq.vals[var].metaparameters = metaparams
            eq.vals[var].weights_internal = np.ones(len(eq.vals[var].structure) - 1)
            eq.vals[var].weights_internal_evald = True
            eq.vals[var].weights_final_evald = True

            _, _, features = eq.vals[var].evaluate(normalize=False, return_val=False)
            assert len(eq.vals[var].weights_final[:-1]) == (features.shape[1]), "Different number of features. Check the structure of the equation."

        return eq

    def compare(self, correct_obj, incorrect_obj, fit_operator, all_vars):
        fit_operator.apply(correct_obj, {})
        fit_operator.apply(incorrect_obj, {})
        return all(
            correct_obj.vals[var].coefficients_stability < incorrect_obj.vals[var].coefficients_stability
            for var in all_vars
        )


class SystemComparison(ComparisonStrategy):
    def _create_eq(self, eq_str, target_var, base_pool, all_vars):
        families_copy = [copy.deepcopy(fam) for fam in base_pool.families]
        for fam in families_copy:
            if hasattr(fam, "variable") and fam.variable is not None and fam.variable != target_var:
                fam.status["demands_equation"] = False
        temp_pool = TFPool(families_copy)
        soeq = translate_equation(eq_str, temp_pool, all_vars=[target_var])
        return soeq.vals[target_var]

    def build(self, symbolic_list, search_obj, all_vars):
        metaparams = {("sparsity", var): {"optimizable": False, "value": 1e-6} for var in all_vars}
        eqs = {}
        for var, eq_str in zip(all_vars, symbolic_list):
            eq = self._create_eq(eq_str, var, search_obj.pool, all_vars)
            eq.main_var_to_explain = var
            eq.metaparameters = metaparams
            eq.weights_internal = np.ones(len(eq.structure) - 1)
            eq.weights_internal_evald = True
            eq.weights_final_evald = True
            eqs[var] = eq

        system = SoEq(search_obj.pool, metaparams)
        system.vals = Chromosome(eqs, {})
        system.moeadd_set = True
        return system

    def compare(self, correct_obj, incorrect_obj, fit_operator, all_vars):
        fit_operator.apply(correct_obj, {})
        fit_operator.apply(incorrect_obj, {})
        return all(
            correct_obj.vals[var].coefficients_stability < incorrect_obj.vals[var].coefficients_stability
            for var in all_vars
        )