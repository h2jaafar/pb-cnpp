# pb-cnpp
A repo containing scripts for preparing PathBench json maps for compatibility with CNPP algorithm.

## Acknowledgments
The original source code for this repo is from  [Path Planning Using Deep Learning](https://alexandria.physik3.uni-goettingen.de/cns-group/datasets/path_planning/). 

The paper, [One-shot path planning for multi-agent systems using fully convolutional neural network](https://arxiv.org/abs/2004.00568), presents the algorithm in more detail.

## What is changed
The purpose of this repo is to compare [WPN](https://arxiv.org/abs/2105.00312) to CNPP. To achieve this, I had to train and test using [PathBench](https://arxiv.org/abs/2105.01777) generated maps. The maps are transformed to a compatible format in `json_to_dat.py`. This is the major addition. 

I also implemented some basic metrics, such as success rates, deviation, prediction times, and distance left when failed. 

There is a resources folder that is omitted from the repo due to the size. 
I have a shortened version in as `sample_resources`, which has the basic file structure that is required as inputs. `lengths_house.json` are the path lengths exported from PathBench. Similarly, `paths_house.json` are the Astar paths exported from PathBench. These are used for ground-truth, as well as deviation metrics. 

### Dat files:
The CNPP code takes dat files as inputs. Each row of the dat file is a map, with length of n x n, where n is the size of the map. i.e, a 8x8 map would have a row-length of 64. Obstacles are denoted by 1. 

`g_maps.dat` is the position of the goal, and `s_maps.dat` is the position of the start point. `inputs.dat` is the obstacle map, and `outputs.dat` is the astar paths.


## Trained Model
The model that is trained on PathBench maps is `model_2d_30k_combined_2.hf5`. This was trained on 30,000 64x64 maps. 


## Inference
To infer/test on maps, you can use any one of the `predict_path...` files. The differences were simply for ease of tracking, as I had different experiments setup for WPN comparisions. 

## Future works
There are no plans to continue/add on to this implementation from my end, however, feel free to submit a PR with any changes you think are suited. 