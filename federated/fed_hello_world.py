import tensorflow_federated as tff


def my_function():
    return "Hello, World!"


output = tff.federated_computation(my_function)()
print(output)
