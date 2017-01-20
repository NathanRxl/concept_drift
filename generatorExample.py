from generator import SEALoader, Generator, KDDCupLoader


def example(generate):
    for i, (xi, yi) in enumerate(generate(batch=3, limit=15)):
        print('iteration number {0}'.format(i))
        print(xi)
        print(yi)
# SEA
print('SEA')
loader = SEALoader('../data/sea.data')
generator = Generator(loader)
example(generator.generate)

# KDDCup
print('\nKdd CUP')
kdd_loader = KDDCupLoader('../data/kddcup.data_10_percent')
generator_kdd = Generator(kdd_loader)
example(generator_kdd.generate)


