## Run

To train and test the mode  with following commands:

```sh
sh scripts/train.sh $GPU $lambda1 $Dataset $seed $lr $alpha $beta $m3
sh scripts/test.sh $GPU $lambda1 $Dataset $seed $lr $alpha $beta $m3
```

1. `lambda1` : the coefficient of the node_embedding
2. `Dataset`: {rams|wikievent}
3. `alpha`: the hyperparameter of the co-existence relation
4. `beta` : the hyperparameter of the co-reference relation
