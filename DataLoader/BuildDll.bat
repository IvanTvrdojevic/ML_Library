gcc -c -o bin/DataLoader.o DataLoader.c -Ilib -D DATALOADER_EXPORTS
gcc -o bin/DataLoader.dll bin/DataLoader.o -s -shared -Wl,--subsystem,windows
gcc -o ../Tests/bin/DataLoader.dll bin/DataLoader.o -s -shared -Wl,--subsystem,windows

