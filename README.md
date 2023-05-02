## CMoAN
Source code for paper "A Selection-pattern-aware Recommendation Model with Colored-Motif Attention Network"

## Introduction
  Heterogeneous information networks (HIN) based models are now widely used to fuse auxiliary information for personalized recommendation. However, existing works mainly focus on capturing high-order connections between heterogeneous nodes through predefined pattern (such as metapath, etc.), which depends on relevant domain knowledge and ignores the statistical characteristics of the substructures in HIN. How to capture meaningful personalized behavior patterns, especially the selection patterns (i.e. how a user select an item), and incorporate them into the preference model is still a challenge in the research of HIN-based recommendation. Therefore, in this paper, a specifically-designed Colored-Motif Attention Network (CMoAN) is proposed to deal with this problem. In the proposed CMoAN, colored motif are used as the elemental building blocks to represent context-based selection patterns, and then by constructing a motif-based adjacency matrix to capture higher-order semantic association between nodes in HIN. Besides, an attentive graph neural network is designed to efficiently model the semantics and high-order relations information. Extensive experiments on three real-world datasets demonstrate that CMoAN consistently outperforms state-of-the-art methods. Furthermore, experimental results also verify the effectiveness of using colored motif for capturing users’selection patterns.

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.11.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1
* cython == 0.29.15

## C++ evaluator
We have implemented C++ code to output metrics during and after training, which is much more efficient than python evaluator. It needs to be compiled first using the following command. 

```
python setup.py build_ext --inplace
```

After compilation, the C++ code will run by default instead of Python code.

## DataSet
### movie_0.8_alldata_5

* `train.txt`
  * Train file.
  * Each line is a user with her/his positive interactions with movies: userID\t a list of movieID\n.

* `test.txt`
  * Test file (positive instances).
  * Each line is a user with her/his positive interactions with movis: userID\t a list of movieID\n.
  * Note that here we treat all unobserved interactions as the negative instances when reporting performance.

* `node_info.txt`
  * node type file.
  * Each line is a nodeID with node Type. 0-user 1-movie 2-actor 3-director 4-type

* `original_graph_weight.txt`
  * Describe the connection relationship between users and each movie in the training set.
 
 * `uma_motif_graph_weight.txt`
  * Describe the connection relationship between users, movies, and actors in the semantic graph.
 
 * `umd_motif_graph_weight.txt`
  * Describe the connection relationship between users, movies, and director in the semantic graph.
  
 * `um1t_motif_graph_weight.txt`
  * Describe the connection relationship between users, movies, and type in the semantic graph.
  

## Examples 
* Command
```
python CMoAN_movie.py --dataset movie_0.8_alldata_5 --regs [1e-3] --embed_size 64 --layer_size [64,64] --lr 0.001 --batch_size 1024 --epoch 500
```
Other parameters can be adjusted as needed, refer to `utility/parser.py`
