
import math

def int_to_byte_array(n):
    ba = [0 for i in range(64)]

    for i in range(63, 0, -1):
        ba[i] = n % 2
        n = math.floor(n / 2)

    return ba


def byte_array_to_int(ba):
    n = 0

    for i in range(63, 0, -1):
        power_of_two = 2 ** (63 - i)
        n += ba[i] * power_of_two

    return n