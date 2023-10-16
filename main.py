import math

# 1. Rosenbrock Function
def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

# 2. Beale's Function
def beale(x, y):
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2

# 3. Matyas Function
def matyas(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

# 4. Himmelblau's Function
def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

# 5. Ackley's Function
def ackley(x, y):
    return -20 * math.exp(-0.2 * math.sqrt(0.5 * (x ** 2 + y ** 2))) - \
           math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))) + math.e + 20

# 6. Levy Function
def levy(x, y):
    return math.sin(3 * math.pi * x) ** 2 + (x - 1) ** 2 * (1 + math.sin(3 * math.pi * y) ** 2) + \
           (y - 1) ** 2 * (1 + math.sin(2 * math.pi * y) ** 2)

# 7. Easom Function
def easom(x, y):
    return -math.cos(x) * math.cos(y) * math.exp(-(x - math.pi) ** 2 - (y - math.pi) ** 2)

# 8. Six-Hump Camel Back Function
def six_hump_camel_back(x, y):
    return (4 - 2.1 * x ** 2 + x ** 4 / 3) * x ** 2 + x * y + (-4 + 4 * y ** 2) * y ** 2

# 9. Cross-in-Tray Function
def cross_in_tray(x, y):
    return -0.0001 * (abs(math.sin(x) * math.sin(y) * math.exp(abs(100 - (math.sqrt(x ** 2 + y ** 2) / math.pi)))) + 1) ** 0.1

# 10. Eggholder Function
def eggholder(x, y):
    return -(y + 47) * math.sin(math.sqrt(abs(x / 2 + (y + 47)))) - x * math.sin(math.sqrt(abs(x - (y + 47))))

# 11. Holder Table Function
def holder_table(x, y):
    return -abs(math.sin(x) * math.cos(y) * math.exp(abs(1 - (math.sqrt(x ** 2 + y ** 2) / math.pi))))

# 12. Rastrigin Function
def rastrigin(x, y):
    return 20 + (x ** 2 - 10 * math.cos(2 * math.pi * x)) + (y ** 2 - 10 * math.cos(2 * math.pi * y))

# 13. Schaffer Function N. 2
def schaffer_N2(x, y):
    return 0.5 + ((math.sin(x ** 2 - y ** 2) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2)

# 14. Styblinski-Tang Function
def styblinski_tang(x, y):
    return 0.5 * (x ** 4 - 16 * x ** 2 + 5 * x + y ** 4 - 16 * y ** 2 + 5 * y)

# 15. Bukin Function N. 6
def bukin_N6(x, y):
    return 100 * math.sqrt(abs(y - 0.01 * x ** 2)) + 0.01 * abs(x + 10)

# 16. Drop-Wave Function
def drop_wave(x, y):
    return - (1 + math.cos(12 * math.sqrt(x ** 2 + y ** 2))) / (0.5 * (x ** 2 + y ** 2) + 2)

# 17. Schwefel Function N. 2.21
def schwefel_N221(x, y):
    return max(abs(x), abs(y))

# 18. Schwefel Function N. 2.22
def schwefel_N222(x, y):
    return abs(x) + abs(y) + abs(x * y)

# 19. Schwefel Function N. 2.23
def schwefel_N223(x, y):
    return abs(x ** 2 - y) + abs(x - 2 * y ** 2 + 1)

# 20. Schwefel Function N. 2.26
def schwefel_N226(x, y):
    return 0.01 * (x ** 2 + y ** 2) + math.sin(x ** 2 + y ** 2)

# 21. Levi N.13 Function
def levi_N13(x, y):
    return math.sin(3 * math.pi * x) ** 2 + (x - 1) ** 2 * (1 + math.sin(3 * math.pi * y) ** 2) + \
           (y - 1) ** 2 * (1 + math.sin(2 * math.pi * y) ** 2)

# 22. Dixon-Price Function
def dixon_price(x, y):
    return (x - 1) ** 2 + 2 * (2 * y ** 2 - x) ** 2

# 23. Three-Hump Camel Function
def three_hump_camel(x, y):
    return 2 * x ** 2 - 1.05 * x ** 4 + (x ** 6 / 6) + x * y + y ** 2

# 24. Shubert's Function
def shubert(x, y):
    result = 0
    for i in range(1, 6):
        result += i * math.cos((i + 1) * x + i) * i * math.cos((i + 1) * y + i)
    return result

# 25. Michalewicz Function
def michalewicz(x, y):
    return -math.sin(x) * (math.sin(x ** 2 / math.pi) ** 20) - math.sin(y) * (math.sin(2 * y ** 2 / math.pi))
