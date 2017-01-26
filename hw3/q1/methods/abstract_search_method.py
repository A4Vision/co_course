
class SearchMethod(object):
    def step(self, eta):
        raise NotImplementedError()

    def full_solution(self):
        raise NotImplementedError()

    def score(self):
        return self.full_solution().score()
