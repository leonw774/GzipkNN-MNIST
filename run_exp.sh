for imgformat in png qoi; do
    for use_binary in "" "-b"; do
        for init_k in 2 8 32; do
            python3 main.py -f $imgformat -k $init_k -d 0 $use_binary --cm
            python3 main.py -f $imgformat -k $init_k -d 1 $use_binary --cm
        done
    done
done
