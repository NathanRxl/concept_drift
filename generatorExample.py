from generator import SEALoader, Generator

loader = SEALoader('../data/sea.data')

generator = Generator(loader)


def example(generate):
    for i, (xi, yi) in enumerate(generate(batch=3, limit=15)):
        print('iteration number {0}'.format(i))
        print(xi)
        print(yi)

example(generator.generate)
