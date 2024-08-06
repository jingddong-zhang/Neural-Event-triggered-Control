For calculating the inference time of all the methods, we recommend to use the following code in the original .py documents:
```
import timeit
start = timeit.default_timer()
m = 5 # num. of random seeds

...

end = timeit.default_timer()
print(f'average inference time={(end - start) / m}')
```

Run the Classic_ETC.py, Critic-Actor NN.py, NETC-high.py for generating data in Fig. 1(a),1(b).

Run NETC-high_noise.py for data in Fig. 1(c),1(d).

Run NETC-high_ablation.py for data in Fig. 1(e), NETC-low_ablation.py for data in Fig. 1(f).

Run fig_rebuttal.py for reproducing the Fig. 1.
