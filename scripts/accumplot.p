failglob = 'ls '.dir.'/fail_'.num.'_*'
faillist = system(failglob)
succglob = 'ls '.dir.'/succ_'.num.'_*'
succlist = system(succglob)
set datafile separator ','
unset key
plot for [file in faillist] file using 2:3 w l lw 2 lc rgb "#F0FF0000", for [file in succlist] file using 2:3 w l lw 2 lc rgb "#F80000FF"
