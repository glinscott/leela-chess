BEGIN {
    plane = 110;
    numplanes = 112;
} 
{
    i++;
    if (i == 1) {
        if ($0 != 2) {
            print "Only V2 files supported" > "/dev/stderr"
            exit 1
        }
    }
    if (i == 2) {
        split($0, vals, " ");
        line = "";
        for (i in vals) {
            if (int(((i-1)%(9*numplanes))/9) == (plane-1)) {
                line = line " " vals[i]*99.0;
            } else 
                line = line " " vals[i];
        } 
        print substr(line, 2);
    }
    else
        print $0;
}
