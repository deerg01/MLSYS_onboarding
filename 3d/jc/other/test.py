class A:
    def __init__(self, c):
        self.c = c

class P1:
    def __init__(self, p1) -> None:
        self.p1 = p1

class P2:
    def __init__(self, p2) -> None:
        self.p2 = p2

class C(P1, P2):
    def __init__(self, p1, p2) -> None:
        super().__init__(p1)
        super(P2, self).__init__(p2)

print(C.mro())
a = A(C(111,222))
print(a.c.p1)
print(a.c.p2)