import three;
size(4cm,0);

draw((0,0,0)--(2,0,0), blue); //x-axis
draw((0,0,0)--(0,2,0), green); //y-axis
draw((0,0,0)--(0,0,2), red); 

file fin=input('g_cryst.txt');

int ix, iy, iz, c;
int n = fin;

for (int i=0; i<n; ++i) {
    ix = fin;
    iy = fin;
    iz = fin;
    c = fin;
    if (c == 0) {
    dot((ix, iy, iz), red);
    }
    if (c == 1) {
    dot((ix, iy, iz), green);
    }
    if (c == 2) {
    dot((ix, iy, iz), blue);
    }
}