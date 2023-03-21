glob = 'ls '.dir.'/*.csv'
list = system(glob)
print list
set datafile separator ','
unset key
plot for [file in list] file using 2:3 w l lw 2 lc rgb "#F00000FF"
