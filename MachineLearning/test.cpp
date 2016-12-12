#include <iostream>
#include <stdlib.h>
using namespace std;

int main()
{
    int rand_start = -1;
    for (int j = 0; j < 10; j++)
    {
        rand_start = rand() % 10;
        for (int i = rand_start; i < 10 + rand_start; i++)
            cout << i % 10 << " ";
        cout << endl;
    }
    
}