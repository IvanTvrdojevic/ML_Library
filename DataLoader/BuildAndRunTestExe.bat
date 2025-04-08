gcc -c -o bin/TestDataLoaderDLL.o TestDataLoaderDLL.c   
gcc -o bin/TestDataLoaderDLL.exe -s bin/TestDataLoaderDLL.o -L. -lDataLoader
bin/TestDataLoaderDLL.exe 