def square(x):
    return x * x


def calculate(client, count=5):
    return client.gather(client.map(square, range(count)))


def fail(client):
    raise RuntimeError("intentional workload failure")
