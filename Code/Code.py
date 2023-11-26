from Toolbox import CancerDataset

#Loading dataset
path = "/home/ubuntu/Final_Project/Data/"
dataset = CancerDataset(path)

#EDA
eda = EDA(path)
eda.plot_class_distribution()
eda.plot_sample_images()


