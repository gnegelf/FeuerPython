matlab='/Applications/MATLAB_R2017b.app/bin/matlab'
for i in 10 20 30 40 50 60
do 
$matlab -nodisplay -nojvm -r "addpath('~/MIPDECO/Feuerprojekt');create_instances($i,$i,30,60,0,1); exit;"
python FeuerVertex.py 0 0 $i $i 30 60 10 10
for j in 30 40 50 60
do
rm ~/python/data/contaData${i}_${j}_5.mat
done
done
