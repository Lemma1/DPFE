## Estimating multi-year 24/7 origin-destination demand using high-granular multi-source traffic data


Implemented by Wei Ma, advised by Sean Qian, Civil and environmental engineering, Carnegie Mellon University. 


### Requirements

- Python 2.7.13
- PyTorch 0.2.0_3
- Numpy 1.13.3
- Scipy 0.19.1
- NetworkX 1.11
- pickle
- joblib 0.11
- pandas 0.18.1

### Instructions

Please clone the whole repo, and run DPFE-v0.1.ipynb using jupyter notebook.


### File specifications

- P_matrix: store the route choice portion matrices
- Q_vector: store the estimated dynamic OD
- R_matrix: store the DAR matrices
- X_vector: store the observed link flow
- observe_index_N.npy: observed link indices
- link_count_data.pickle: flow data
- link_spd_data.pickle: speed data
- od_list.pickle: OD information
- graph.pickle: graph information
- cluster_info.pickle: traffic scenario information
- base.py: data processing, DAR matrix construction, P matrix construction
- pfe.py: stochastic projected gradient descent
- DPFE-v0.1.ipynb: main script, start from here


### Paper
[Estimating multi-year 24/7 origin-destination demand using high-granular multi-source traffic data](https://www.sciencedirect.com/science/article/pii/S0968090X18302948)

### Data

Since the traffic speed data (link_spd_data.pickle) and count data (link_count_data.pickle) are under the non-discloure agreement, please contact the authors to obtain the data.


For any questions, please contact Lemma171@gmail.com

