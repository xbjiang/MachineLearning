#include <iostream>
#include <stdio.h>
using namespace std;
class GamePlayer
{
private:
    static const int NumTurns = 5;
    int scores[NumTurns];
};

//const int GamePlayer::NumTurns;

char* p = "Hello World1";
char* p1 = "Hello World1";
char c[] = "Hello World2";

int main()
{
    //p[1] = 'E';
    //c[1] = 'E';
    printf("%d\n", p);
    printf("%d\n", p1);
}