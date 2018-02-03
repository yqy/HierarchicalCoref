python ana_pair_learning.py -gpu 2 > result.ana
python ana_top_pair_learning.py -gpu 2 > result.top.ana
python pair_learning.py -gpu 2 > result
python top_pair_learning.py -gpu 2 > result.top
python reinforce_learning.py -gpu 2 > result.rl
cat results.rl | python read_result.py > result.best
