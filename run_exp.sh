for imgformat in png qoi; do
    for init_k in 2 8 32; do
        for concat_dim in 0 1; do
            python3 main.py -f $imgformat -k $init_k -d $concat_dim --cm
        done
    done
done