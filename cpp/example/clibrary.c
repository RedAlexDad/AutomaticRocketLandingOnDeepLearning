#include <stdio.h>

void prompt(int num) {
    printf("The number %d was entered", num);
}

// Выполните это в команде
// gcc -fPIC -shared -o clibrary.so clibrary.c