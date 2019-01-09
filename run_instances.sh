matlab='/Applications/MATLAB_R2017b.app/bin/matlab'
for i in 3 40 50 60 70 80
do 
matlab -nodisplay -nojvm -r "addpath('~/MIPDECO/Feuerprojekt');create_instances($i,$i,30,60,0,1); exit;"
python FeuerVertex.py 0 0 $i $i 30 60
for j in 30 40 50 60
do
rm ~/python/data/feuerData${i}_${j}_2.mat
rm -r ~/MIPDECO/Feuerprojekt/data/matrix*
done
done
