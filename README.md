# cigChannel
Create synthetic seismic volumes with labeled meandering channels, tributary channel networks, and submarine canyons.

<img src="https://github.com/user-attachments/assets/65502674-7252-44ef-8931-2ef5003d9bc0" alt="Alt Text" style="width:50%; height:auto;">

See our ESSD paper for details:  
[cigChannel: a large-scale 3D seismic dataset with labeled paleochannels for advancing deep learning in seismic interpretation](https://doi.org/10.5194/essd-17-3447-2025)

The cigChannel dataset, containing 1600 seismic volumes, is available [here](https://doi.org/10.5281/zenodo.10791151)

## Installation
1. Clone this repository to your device:\
`git clone https://github.com/wanggy-1/cigChannel.git`
2. Install all the required packages by running the following command:\
`conda env create -f env.yml`  
 or prepare an environment with python >= 3.10, and then run the following command:\
`pip install -r requirements.txt`

## Customize your own seismic volumes
We have provided demonstration codes for your refenrence to create your own seismic volumes.

### Seismic volume with meandering channels
Run `meandering_channel.py`, and you will get a seismic impedance model, a seismic volume, and a channel label volume.\
<img src="https://github.com/user-attachments/assets/b3dc017a-1c75-44a9-b3bb-e1bbfab28335" alt="meandering" style="width:80%; height:auto;" >\
You can also create multiple seismic volumes in parallel following these steps:
1. Download the zip file of channel distance maps from [here](https://huggingface.co/datasets/wangguangyu/cigChannel_building_material/blob/main/Distmap.zip), which contains thousands of prefabricated meandering channels.
2. Unzip it to `./Distmap`.
3. Run `meandering_channel_parallel.py`. It accelerates the process of data generation by creating meandering channels from those distance maps.

### Seismic volume with tributary channel networks.
Download the zip file of channel topography maps from [here](https://huggingface.co/datasets/wangguangyu/cigChannel_building_material/blob/main/Topography.zip), which contains thousands of prefabricated tributary channel networks.\
Unzip it to `./Topography`.
Run `tributary_channel.py`. It creates tributary channel networks from the topography maps.\
You will get a seismic impedance model, a seismic volume, and a channel label volume.\
<img src="https://github.com/user-attachments/assets/d5eb897d-a5e0-4f49-b5d0-415f63655391" alt="tributary" style="width:80%; height:auto;">\
You can also create multiple seismic volumes in parallel by running `tributary_channel_parallel.py`

### Seismic volume with submarine canyons.
Run `submarine_canyon.py`, and you will get a sedimentary facies model, a seismic impedance model, a seismic volume, and a channel label volume.\
<img src="https://github.com/user-attachments/assets/53fb8b19-7eb7-4861-9c32-8fe7e6e2ae94" alt="submarine" style="width:80%; height:auto;">\
In the sedimentary facies model, 2 represents point-bars (yellow), 3 represents natural levees (dark gold), 4 represents abandoned meanders (saddle brown), and 0 represents the background (white).\
You can also create multiple seismic volumes in parallel by running `submarine_channel_parallel.py`

### Seismic volume with assorted channels.
Download the zip file of [channel distance maps](https://huggingface.co/datasets/wangguangyu/cigChannel_building_material/blob/main/Distmap.zip) and [channel topography maps](https://huggingface.co/datasets/wangguangyu/cigChannel_building_material/blob/main/Topography.zip).\
Unzip them to `./Distmap` and `./Topography`.\
Run `assorted_channel.py`, and you will get a sedimentary facies model, a seismic impedance model, a seismic volume, and a channel label volume.\
<img src="https://github.com/user-attachments/assets/f355a16a-180e-48b1-9bf7-1e88304ae034" alt="assorted" style="width:80%; height:auto;">\
In the sedimentary facies model, 1 represents channel lag deposits (green), 2 represents point-bars (yellow), 3 represents natural levees (dark gold), 4 represents abandoned meanders (saddle brown), and 0 represents the background (white).\
You can also create multiple seismic volumes in parallel by running `assorted_channel_parallel.py`

## AI models for channel identification
Tutorial coming up.
