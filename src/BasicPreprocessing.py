class BasicPreprocessing:
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path
        self.image_size = 128
        self.batch_size = 32
        self.train_ratio = None
        self.val_ratio = None
        self.test_ratio = None
        self.random_seed = 42
        

    def import_dataset(self):  #to load dataset and to store class names
        pass

    def inspect_dataset(self): #print class count and dataset size
        pass

    def transforms(self):  # to define train transforms and validation and train transforms
        pass

    def split_dataset(self): # to split the dataset between train/validation and test
        pass

    def create_dataloaders(self): # wrap the splits
        pass

    def visualize(self):  # to visualize transformed images
        pass