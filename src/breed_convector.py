class BreedConvector:
    def __init__(self):
        self.dirname_to_index = {'n02086240': 0,
                                 'n02087394': 1,
                                 'n02088364': 2,
                                 'n02089973': 3,
                                 'n02093754': 4,
                                 'n02096294': 5,
                                 'n02099601': 6,
                                 'n02105641': 7,
                                 'n02111889': 8,
                                 'n02115641': 9}
        self.index_to_breed = {0: 'Shih-Tzu',
                               1: 'Rhodesian ridgeback',
                               2: 'Beagle',
                               3: 'English foxhound',
                               4: 'Border terrier',
                               5: 'Australian terrier',
                               6: 'Golden retriever',
                               7: 'Old English sheepdog',
                               8: 'Samoyed',
                               9: 'Dingo'}

        self.index_to_dirname = {val: key for key, val in self.dirname_to_index.items()}
        self.breed_to_index = {val: key for key, val in self.index_to_breed.items()}
