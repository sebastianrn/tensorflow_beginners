import numpy as np


class Vector(object):
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(coordinates)
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')

    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)

    def __eq__(self, v):
        return self.coordinates == v.coordinates

    def addition(self, v):
        new_coordinates = [x + y for x, y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

    def substraction(self, v):
        new_coordinates = [x - y for x, y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

    def scalarMultiplication(self, c):
        new_coordinates = [c * x for x in self.coordinates]
        return Vector(new_coordinates)

    def linearFunction(v1, v2, b):
        vector_product = [x * y for x, y in zip(v1.coordinates, v2.coordinates)]
        result = np.sum(vector_product) + b
        return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


values = Vector([-4, 5])
weight = Vector([4, 5])
bias = -9

score = weight.linearFunction(values, bias)
print("Score: ", score)
print("Sigmoid Value: ", sigmoid(score))
