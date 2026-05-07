from abc import ABC, abstractmethod
from tests.functional.utils import Timer

class EquationTestTemplate(ABC):
    strategy = None

    @abstractmethod
    def make_search(self):
        pass

    @abstractmethod
    def correct_symbolic(self):
        pass

    @abstractmethod
    def incorrect_symbolic(self):
        pass

    @abstractmethod
    def all_vars(self):
        pass

    def run(self, fit_operator, do_discovery=False):
        search_obj = self.make_search()

        if do_discovery:
            return self.run_discovery(search_obj)

        correct_obj = self.strategy.build(self.correct_symbolic(), search_obj, self.all_vars())
        incorrect_obj = self.strategy.build(self.incorrect_symbolic(), search_obj, self.all_vars())

        with Timer() as t:
            ok = self.strategy.compare(correct_obj, incorrect_obj, fit_operator, self.all_vars())

        return ok, t.elapsed


    def run_discovery(self, search_obj):
        raise NotImplementedError