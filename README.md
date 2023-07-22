# UHRCT_SR
Here is the project responded MICCAI 2023: "Topology-Preserving Computed Tomography Super-resolution Based on Dual-stream Diffusion Model". The data and code will be released as soon as they are been well organized. 

# Dataset
Here we share the ultra-high-resolution (UHRCT) CT scans (1024*1024) with 1.00 mm thickness. The link of the dataset is https://drive.google.com/drive/folders/1fKHu7mE5fKxknrjy6wibL5prTb8_9GV0?usp=sharing. The shared dataset consists of 95 URCT datasets in the format of .npz. You can use np.load(path_to_your_dataset)["arr_0"] to open them. Some of the CT scans are cliped to [-1000, 600].

