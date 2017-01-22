from DataLoader import SEALoader, KDDCupLoader
from StreamGenerator import StreamGenerator


def stream_example(stream_generator):
    for i, (Xi, yi) in enumerate(stream_generator.generate(batch=3, stream_length=15)):
        print('\t\nIteration #{0}'.format(i))
        print('\t\tX', Xi)
        print('\t\ty', yi)

# SEA
print('SEA stream')
sea_data_loader = SEALoader('data/sea.data')
sea_stream_generator = StreamGenerator(sea_data_loader)
stream_example(sea_stream_generator)

# KDDCup
print('Kdd CUP')
kdd_data_loader = KDDCupLoader('data/kddcup.data_10_percent')
kdd_stream_generator = StreamGenerator(kdd_data_loader)
stream_example(kdd_stream_generator)
