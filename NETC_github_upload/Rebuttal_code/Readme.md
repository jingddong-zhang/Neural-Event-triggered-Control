For calculating the inference time of all the methods, we recommend to use the following code in the original .py documents:
```
import timeit
start = timeit.default_timer()
m = 5 # num. of random seeds

...

end = timeit.default_timer()
print(f'average inference time={(end - start) / m}')
```
